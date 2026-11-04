#pragma once
#include "ops/linear/kernels/volta/qpn.cuh"
#include <mma.h>

namespace ninfer::ops::detail {
// Prefill reuses each staged weight tile over 32 tokens. Staging stays within
// the CTA; the public output and workspace contracts are unchanged.
template <bool Paired, class Operands, class Output, class Epilogue>
__global__ __launch_bounds__(128, 4) void volta_tiled_kernel(Operands p, Output output,
                                                             Epilogue epilogue) {
    __shared__ __align__(32) half xs[2][32][40];
    __shared__ __align__(32) half ws[2][32][40];
    __shared__ __align__(32) float result[32][32];
    __shared__ unsigned maxima[4];
    __shared__ VoltaActivationResiduals residuals;
    const int token0             = int(blockIdx.y) * 32;
    const int tokens             = min(32, p.tokens - token0);
    const auto* x                = p.x + std::int64_t(token0) * p.k;
    const float activation_scale = volta_activation_scale(x, tokens * p.k, maxima);
    if (volta_activation_residuals(x, tokens * p.k, activation_scale, residuals)) {
        for (int e = threadIdx.x; e < 1024; e += 128) {
            const int row = volta_qpn_row<Paired>(e / 32, p.rows), token = e % 32;
            float value = 0;
            if (row < p.rows && token < tokens) {
                for (int k = 0; k < p.k; ++k)
                    value = fmaf(__bfloat162float(x[std::int64_t(token) * p.k + k]),
                                 volta_weight(p, row, k), value);
                value *= volta_weight_factor(p, row);
            }
            result[e / 32][token] = value;
        }
    } else {
        const int warp = threadIdx.x / 32;
        using namespace nvcuda;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.f);
        // Each thread loads eight adjacent K values, with four threads per row.
        const int local_row = threadIdx.x / 4, local_k = (threadIdx.x % 4) * 8;
        const int weight_row = volta_qpn_row<Paired>(local_row, p.rows);

        struct Carry {
            half xv[8], wv[8];
        } carry;

        const auto prefetch = [&](int kbase) {
            uint4 raw = {};
            if (local_row < tokens)
                raw = __ldg(reinterpret_cast<const uint4*>(x + std::int64_t(local_row) * p.k +
                                                           kbase + local_k));
            const auto* activation = reinterpret_cast<const __nv_bfloat16*>(&raw);
#pragma unroll
            for (int j = 0; j < 8; ++j)
                carry.xv[j] = __float2half_rn(__bfloat162float(activation[j]) * activation_scale);
            const int safe_row = min(weight_row, p.rows - 1);
            if constexpr (std::is_same_v<Operands, Nvfp4A16Operands>) {
                const unsigned word = __ldg(reinterpret_cast<const unsigned*>(
                    p.codes + std::int64_t(safe_row) * (p.k / 2) + (kbase + local_k) / 2));
                const float scale   = common::volta::decode<common::volta::Format::E4m3>(
                    p.scales[volta_nvfp4_scale(safe_row, (kbase + local_k) / 16, p.k)]);
                const half2 rebias = __float2half2_rn(16384.f), sc = __float2half2_rn(scale);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const unsigned bits =
                        ((word << (12 - 4 * j)) & 0x80008000u) |
                        (j < 3 ? (word << (9 - 4 * j)) & 0x0e000e00u : (word >> 3) & 0x0e000e00u);
                    const half2 decoded =
                        __hmul2(__hmul2(*reinterpret_cast<const half2*>(&bits), rebias), sc);
                    carry.wv[j]     = __low2half(decoded);
                    carry.wv[j + 4] = __high2half(decoded);
                }
            } else {
                const uint2 codes = __ldg(reinterpret_cast<const uint2*>(
                    p.codes + std::int64_t(safe_row) * p.k + kbase + local_k));
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const unsigned word = j < 4 ? codes.x : codes.y;
                    carry.wv[j] =
                        __float2half_rn(common::volta::decode<common::volta::Format::E4m3>(
                            (word >> ((j % 4) * 8)) & 255));
                }
            }
        };
        const auto commit = [&](int buf) {
            *reinterpret_cast<uint4*>(&xs[buf][local_row][local_k]) =
                *reinterpret_cast<uint4*>(carry.xv);
            *reinterpret_cast<uint4*>(&ws[buf][local_row][local_k]) =
                *reinterpret_cast<uint4*>(carry.wv);
        };
        prefetch(0);
        commit(0);
        int buf = 0;
        for (int kbase = 0; kbase < p.k; kbase += 32) {
            __syncthreads();
            const bool next = kbase + 32 < p.k;
            if (next) prefetch(kbase + 32);
#pragma unroll
            for (int kk = 0; kk < 32; kk += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
                wmma::load_matrix_sync(a, &ws[buf][(warp / 2) * 16][kk], 40);
                wmma::load_matrix_sync(b, &xs[buf][(warp % 2) * 16][kk], 40);
                wmma::mma_sync(acc, a, b, acc);
            }
            if (next) commit(buf ^ 1);
            buf ^= 1;
        }
        wmma::store_matrix_sync(&result[(warp / 2) * 16][(warp % 2) * 16], acc, 32,
                                wmma::mem_row_major);
        __syncthreads();
        for (int e = threadIdx.x; e < 1024; e += 128) {
            const int row = volta_qpn_row<Paired>(e / 32, p.rows);
            if (row < p.rows)
                result[e / 32][e % 32] =
                    volta_correct(p, row, e % 32, result[e / 32][e % 32] / activation_scale,
                                  residuals) *
                    volta_weight_factor(p, row);
        }
    }
    __syncthreads();
    for (int e = threadIdx.x; e < 1024; e += 128) {
        const int local_row = e / 32, token = e % 32;
        const int row = volta_qpn_row<Paired>(local_row, p.rows);
        if (row < p.rows && token < tokens) {
            if constexpr (Paired) {
                if (local_row < 16)
                    epilogue.apply_pair(output, row, token0 + token, result[local_row][token],
                                        result[local_row + 16][token]);
            } else
                output.store(row, token0 + token,
                             epilogue.apply(row, token0 + token, result[local_row][token]));
        }
    }
}

template <bool Paired = false, class Operands, class Output, class Epilogue>
void launch_volta_tiled(Operands p, Output output, Epilogue epilogue, cudaStream_t stream) {
    const dim3 grid((p.rows + 31) / 32, (p.tokens + 31) / 32);
    volta_tiled_kernel<Paired><<<grid, 128, 0, stream>>>(p, output, epilogue);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail

namespace ninfer::ops::detail {
template <int SmallTiles, int MatrixThreshold, bool Paired = false, class Operands, class Output,
          class Epilogue>
void launch_volta_linear(Operands p, Output output, Epilogue epilogue, cudaStream_t stream) {
    if (p.tokens < MatrixThreshold)
        launch_volta_qpn<SmallTiles, Paired>(p, output, epilogue, stream);
    else
        launch_volta_tiled<Paired>(p, output, epilogue, stream);
}
} // namespace ninfer::ops::detail
