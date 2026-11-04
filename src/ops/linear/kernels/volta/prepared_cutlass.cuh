#pragma once
#include "ops/linear/kernels/volta/prepared_activation.cuh"
#include "cutlass/gemm/device/gemm.h"
#include <stdexcept>

namespace ninfer::ops::detail {

template <class Operands>
__global__ void volta_decode_weight(Operands p, half* weights) {
    const int v = int(blockIdx.x) * 256 + threadIdx.x;
    if (v >= p.rows * (p.k / 8)) return;
    const int row = v / (p.k / 8), k = (v % (p.k / 8)) * 8;
    __align__(16) half result[8];
    if constexpr (std::is_same_v<Operands, Nvfp4A16Operands>) {
        const unsigned word = __ldg(
            reinterpret_cast<const unsigned*>(p.codes + std::int64_t(row) * (p.k / 2) + k / 2));
        const unsigned code = p.scales[volta_nvfp4_scale(row, k / 16, p.k)];
        const half2 sc      = volta_fp8_pair(code, code);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned bits =
                ((word << (12 - 4 * j)) & 0x80008000u) |
                (j < 3 ? (word << (9 - 4 * j)) & 0x0e000e00u : (word >> 3) & 0x0e000e00u);
            const half2 val = __hmul2(
                __hmul2(*reinterpret_cast<const half2*>(&bits), __float2half2_rn(16384.f)), sc);
            result[j]     = __low2half(val);
            result[j + 4] = __high2half(val);
        }
    } else {
        const uint2 words =
            __ldg(reinterpret_cast<const uint2*>(p.codes + std::int64_t(row) * p.k + k));
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const half2 val = volta_fp8_pair(words.x >> (j * 8), words.y >> (j * 8));
            result[j]       = __low2half(val);
            result[j + 4]   = __high2half(val);
        }
    }
    reinterpret_cast<uint4*>(weights)[v] = *reinterpret_cast<const uint4*>(result);
}

template <class Operands>
__global__ void volta_decode_packed_weight(Operands p, half* weights) {
    __shared__ __align__(16) half tile[32][136];
    const int row_block = int(blockIdx.x) / (p.k / 128);
    const int k_block   = int(blockIdx.x) % (p.k / 128);
    const int i         = row_block * 32 * (p.k / 16) + (k_block * 8 + int(threadIdx.x) / 32) * 32 +
                  int(threadIdx.x) % 32;
    const int lane = i % 32, group = (i / 32) % (p.k / 16);
    const int row = i / (32 * (p.k / 16)) * 32 + (lane / 4 % 4) * 8 + lane % 4 + (lane / 16) * 4;
    __align__(16) half2 result[8];
    if constexpr (volta_nvfp4_v<Operands>) {
        const uint2 raw     = __ldg(reinterpret_cast<const uint2*>(p.codes + std::int64_t(i) * 8));
        const unsigned code = p.scales[i];
        const half2 scale   = volta_fp8_pair(code, code);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const unsigned word = j < 4 ? raw.x : raw.y;
            const int shift     = j % 4;
            const unsigned bits =
                ((word << (12 - 4 * shift)) & 0x80008000u) |
                (shift < 3 ? (word << (9 - 4 * shift)) & 0x0e000e00u : (word >> 3) & 0x0e000e00u);
            result[j] = __hmul2(
                __hmul2(*reinterpret_cast<const half2*>(&bits), __float2half2_rn(16384.f)), scale);
        }
    } else {
        const uint4 raw   = __ldg(reinterpret_cast<const uint4*>(p.codes + std::int64_t(i) * 16));
        const auto* words = reinterpret_cast<const unsigned*>(&raw);
#pragma unroll
        for (int j = 0; j < 8; ++j)
            result[j] =
                volta_fp8_pair(words[j / 2] >> (j % 2 * 16), words[j / 2] >> (j % 2 * 16 + 8));
    }
    auto* dst = reinterpret_cast<uint4*>(&tile[row % 32][(group % 8) * 16]);
    dst[0]    = reinterpret_cast<const uint4*>(result)[0];
    dst[1]    = reinterpret_cast<const uint4*>(result)[1];
    __syncthreads();
#pragma unroll
    for (int item = 0; item < 2; ++item) {
        const int linear = int(threadIdx.x) * 8 + item * 2048;
        const int r = linear / 128, c = linear % 128;
        *reinterpret_cast<uint4*>(weights + std::int64_t(row_block * 32 + r) * p.k + k_block * 128 +
                                  c) = *reinterpret_cast<const uint4*>(&tile[r][c]);
    }
}

template <class Operands>
__device__ float volta_finish_value(Operands p, VoltaPreparedActivation prepared,
                                    const float* values, int row, int token) {
    const auto& tile = prepared.tiles[token];
    float value      = values[std::int64_t(token) * p.rows + row] * volta_inverse_scale(tile.scale);
    value            = volta_correct(p, row, 0, value, tile.residuals);
    return value * volta_weight_factor(p, row);
}

// Exceptional rows keep full BF16 range. A warp cooperates on each dot product
// so a dense correction does not serialize the entire K dimension in one thread.
template <class Operands>
__device__ float volta_fallback_dot(Operands p, int row, int token) {
    float value = 0.f;
    for (int k = threadIdx.x % 32; k < p.k; k += 32)
        value = fmaf(__bfloat162float(p.x[std::int64_t(token) * p.k + k]), volta_weight(p, row, k),
                     value);
    for (int step = 16; step; step /= 2) value += __shfl_down_sync(0xffffffffu, value, step);
    return value * volta_weight_factor(p, row);
}

template <bool Paired, class Operands, class Output, class Epilogue>
__global__ void volta_finish_gemm(Operands p, VoltaPreparedActivation prepared, const float* values,
                                  Output output, Epilogue epilogue) {
    constexpr int Items = Paired ? 1 : 8;
    const int token     = blockIdx.y;
    const int rows      = Paired ? p.rows / 2 : p.rows;
    if (prepared.tiles[token].fallback) {
        for (int row = int(blockIdx.x) * (256 * Items) + threadIdx.x / 32;
             row < min(rows, int(blockIdx.x + 1) * (256 * Items)); row += 8) {
            const float value = volta_fallback_dot(p, row, token);
            if constexpr (Paired) {
                const float up = volta_fallback_dot(p, row + rows, token);
                if (threadIdx.x % 32 == 0) epilogue.apply_pair(output, row, token, value, up);
            } else if (threadIdx.x % 32 == 0)
                output.store(row, token, epilogue.apply(row, token, value));
        }
        return;
    }
    for (int item = 0; item < Items; ++item) {
        const int row = int(blockIdx.x) * (256 * Items) + threadIdx.x + item * 256;
        if (row >= rows) return;
        const float value = volta_finish_value(p, prepared, values, row, token);
        if constexpr (Paired)
            epilogue.apply_pair(output, row, token, value,
                                volta_finish_value(p, prepared, values, row + rows, token));
        else
            output.store(row, token, epilogue.apply(row, token, value));
    }
}

template <int TileN>
using VoltaGemm = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<128, TileN, 32>, cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

template <bool Paired, class Operands, class Output, class Epilogue>
void launch_volta_prepared_gemm(Operands p, Output output, Epilogue epilogue,
                                VoltaPreparedActivation prepared, half* weight, float* values,
                                cudaStream_t stream) {
    volta_prepare_activation<1><<<p.tokens, 128, 0, stream>>>(p.x, p.k, p.tokens, prepared);
    if constexpr (volta_packed_v<Operands>)
        volta_decode_packed_weight<<<(p.rows * (p.k / 16) + 255) / 256, 256, 0, stream>>>(p,
                                                                                          weight);
    else
        volta_decode_weight<<<(p.rows * (p.k / 8) + 255) / 256, 256, 0, stream>>>(p, weight);
    using Kernel = VoltaGemm<256>;
    typename Kernel::Arguments args{{p.tokens, p.rows, p.k},
                                    {reinterpret_cast<cutlass::half_t*>(prepared.values), p.k},
                                    {reinterpret_cast<cutlass::half_t*>(weight), p.k},
                                    {values, p.rows},
                                    {values, p.rows},
                                    {1.f, 0.f},
                                    1};
    Kernel gemm;
    if (gemm(args, nullptr, stream) != cutlass::Status::kSuccess)
        throw std::runtime_error("Volta GEMM launch failed");
    const int rows      = Paired ? p.rows / 2 : p.rows;
    constexpr int Items = Paired ? 1 : 8;
    volta_finish_gemm<Paired>
        <<<dim3((rows + 256 * Items - 1) / (256 * Items), p.tokens), 256, 0, stream>>>(
            p, prepared, values, output, epilogue);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail
