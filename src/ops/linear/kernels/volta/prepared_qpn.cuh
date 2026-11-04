#pragma once
#include "ops/linear/kernels/volta/prepared_activation.cuh"

namespace ninfer::ops::detail {
template <int Tiles, int Warps, int Accumulators, bool Paired, class Operands, class Output,
          class Epilogue>
__global__ __launch_bounds__(Warps * 32,
                             2) void volta_prepared_qpn(Operands p, Output output,
                                                        Epilogue epilogue,
                                                        VoltaPreparedActivation prepared) {
    constexpr int Threads = Warps * 32;
    __shared__ float partial[Warps][Tiles * 8 * 32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    const int qp     = (lane / 4) % 4;
    const int r      = lane % 4 + (lane / 16) * 4;
    const int token0 = int(blockIdx.y) * (Tiles * 8);
    const int tokens = min(Tiles * 8, p.tokens - token0);
    const auto* x    = p.x + std::int64_t(token0) * p.k;
    {
        const int weight_row            = volta_qpn_row<Paired>(qp * 8 + r, p.rows);
        const int safe_row              = min(weight_row, p.rows - 1);
        float c[Tiles][Accumulators][8] = {};
        constexpr int Chunk             = volta_packed_v<Operands> ? 16 : 64;
        const int chunks                = p.k / Chunk;
        const int begin                 = warp * (chunks / Warps);
        const int end                   = warp == Warps - 1 ? chunks : begin + chunks / Warps;
        // Unroll the packed decode loop where the extra live values fit without
        // spills. Two-tile NVFP4 batches retain the smaller register footprint.
        constexpr int Unroll = volta_packed_v<Operands> &&
                                       (!volta_nvfp4_v<Operands> || Tiles == 1)
                                   ? 4
                                   : 1;
#pragma unroll Unroll
        for (int chunk = begin; chunk < end; ++chunk) {
            uint4 codes[volta_nvfp4_v<Operands> ? 2 : 4];
            unsigned scales = 0;
            if constexpr (volta_packed_v<Operands>) {
                const auto offset = volta_qpn_fragment(safe_row, chunk, p.k);
                if constexpr (volta_nvfp4_v<Operands>) {
                    const uint2 raw = __ldg(reinterpret_cast<const uint2*>(p.codes + offset * 8));
                    codes[0].x      = raw.x;
                    codes[0].y      = raw.y;
                    scales          = p.scales[offset];
                } else
                    codes[0] = __ldg(reinterpret_cast<const uint4*>(p.codes + offset * 16));
            } else {
#pragma unroll
                for (int i = 0; i < sizeof(codes) / sizeof(uint4); ++i) {
                    constexpr int bytes = volta_nvfp4_v<Operands> ? 32 : 64;
                    codes[i]            = __ldg(reinterpret_cast<const uint4*>(
                        p.codes +
                        std::int64_t(safe_row) * (p.k / (volta_nvfp4_v<Operands> ? 2 : 1)) +
                        chunk * bytes + i * 16));
                }
                if constexpr (volta_nvfp4_v<Operands>)
                    scales = __ldg(reinterpret_cast<const unsigned*>(
                        p.scales + volta_nvfp4_scale(safe_row, chunk * 4, p.k)));
            }
#pragma unroll
            for (int g = 0; g < Chunk / 16; ++g) {
                const int group = chunk * (Chunk / 16) + g;
                half2 b[8];
                const unsigned* words = reinterpret_cast<const unsigned*>(codes);
                if constexpr (volta_nvfp4_v<Operands>) {
                    const unsigned code = (scales >> (g * 8)) & 255;
                    const half2 sc      = volta_fp8_pair(code, code);
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const unsigned word = words[g * 2 + j / 4];
                        const int shift     = j % 4;
                        const unsigned bits = ((word << (12 - 4 * shift)) & 0x80008000u) |
                                              (shift < 3 ? (word << (9 - 4 * shift)) & 0x0e000e00u
                                                         : (word >> 3) & 0x0e000e00u);
                        b[j] = __hmul2(__hmul2(*reinterpret_cast<const half2*>(&bits),
                                               __float2half2_rn(16384.f)),
                                       sc);
                    }
                } else if constexpr (volta_packed_v<Operands>) {
#pragma unroll
                    for (int j = 0; j < 8; ++j)
                        b[j] = volta_fp8_unscaled(words[j / 2] >> ((j % 2) * 16),
                                                  words[j / 2] >> ((j % 2) * 16 + 8));
                } else {
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int first = (j / 4) * 8 + j % 4;
                        b[j] = volta_fp8_pair(words[g * 4 + first / 4] >> ((first % 4) * 8),
                                              words[g * 4 + (first + 4) / 4] >> ((first % 4) * 8));
                    }
                }
#pragma unroll
                for (int tile = 0; tile < Tiles; ++tile) {
                    half2 a[8];
                    const int token = tile * 8 + r;
                    {
                        uint4 raw[2] = {};
                        if (token < tokens) {
                            const auto* source =
                                prepared.values + std::int64_t(token0 + token) * p.k + group * 16;
                            raw[0] = __ldg(reinterpret_cast<const uint4*>(source));
                            raw[1] = __ldg(reinterpret_cast<const uint4*>(source + 8));
                        }
                        const auto* words = reinterpret_cast<const unsigned*>(raw);
                        if constexpr (volta_packed_v<Operands>) {
#pragma unroll
                            for (int j = 0; j < 8; ++j)
                                reinterpret_cast<unsigned*>(a)[j] = words[j];
                        } else
#pragma unroll
                            for (int j = 0; j < 8; j += 4) {
                                auto* pairs = reinterpret_cast<unsigned*>(a + j);
                                pairs[0]    = __byte_perm(words[j], words[j + 2], 0x5410);
                                pairs[1]    = __byte_perm(words[j], words[j + 2], 0x7632);
                                pairs[2]    = __byte_perm(words[j + 1], words[j + 3], 0x5410);
                                pairs[3]    = __byte_perm(words[j + 1], words[j + 3], 0x7632);
                            }
                    }
                    const auto* av = reinterpret_cast<const unsigned*>(a);
                    const auto* bv = reinterpret_cast<const unsigned*>(b);
#pragma unroll
                    for (int j = 0; j < 8; j += 2)
                        volta_qpn_mma(c[tile][(j / 2) % Accumulators], av[j], av[j + 1], bv[j],
                                      bv[j + 1]);
                }
            }
        }
#pragma unroll
        for (int tile = 0; tile < Tiles; ++tile)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int token = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
                const int col   = (i & 1) | (lane & 2) | ((i >> 2) << 2);
                float sum       = 0;
#pragma unroll
                for (int a = 0; a < Accumulators; ++a) sum += c[tile][a][i];
                partial[warp][(tile * 8 + token) * 32 + qp * 8 + col] = sum;
            }
        __syncthreads();
        for (int e = threadIdx.x; e < tokens * 32; e += Threads) {
            float value = 0;
#pragma unroll
            for (int w = 0; w < Warps; ++w) value += partial[w][e];
            if constexpr (std::is_same_v<Operands, VoltaPackedFp8>) value *= 256.f;
            const int row    = volta_qpn_row<Paired>(e % 32, p.rows);
            const auto& tile = prepared.tiles[token0 + e / 32];
            partial[0][e]    = row < p.rows && !tile.fallback
                                   ? volta_correct(p, row, 0, value * volta_inverse_scale(tile.scale),
                                                   tile.residuals) *
                                      volta_weight_factor(p, row)
                                   : 0.f;
        }
    }
    const bool exceptional =
        __syncthreads_or(threadIdx.x < tokens && prepared.tiles[token0 + threadIdx.x].fallback);
    // Handle exceptional activation rows independently; one difficult token must
    // not force all eight queries through a serial FP32 contraction.
    if (exceptional)
        for (int e = warp; e < tokens * 32; e += Warps) {
            if (!prepared.tiles[token0 + e / 32].fallback) continue;
            const int row = volta_qpn_row<Paired>(e % 32, p.rows);
            float value   = 0.f;
            if (row < p.rows) {
                for (int k = lane; k < p.k; k += 32)
                    value = fmaf(__bfloat162float(x[std::int64_t(e / 32) * p.k + k]),
                                 volta_weight(p, row, k), value);
                for (int step = 16; step; step /= 2)
                    value += __shfl_down_sync(0xffffffffu, value, step);
                value *= volta_weight_factor(p, row);
            }
            if (lane == 0) partial[0][e] = value;
        }
    __syncthreads();
    for (int e = threadIdx.x; e < tokens * 32; e += Threads) {
        const int local = e % 32, token = token0 + e / 32;
        const int row = volta_qpn_row<Paired>(local, p.rows);
        if (row < p.rows) {
            if constexpr (Paired) {
                if (local < 16)
                    epilogue.apply_pair(output, row, token, partial[0][e], partial[0][e + 16]);
            } else
                output.store(row, token, epilogue.apply(row, token, partial[0][e]));
        }
    }
}


} // namespace ninfer::ops::detail
