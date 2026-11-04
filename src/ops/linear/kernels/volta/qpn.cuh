#pragma once

#include "core/device.h"
#include "ops/common/cuda/volta_mma.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/kernels/fp8/fp8_operands.h"
#include "ops/linear/kernels/nvfp4/nvfp4_operands.h"

#include <type_traits>

namespace ninfer::ops::detail {

// Native Volta quadpairs: each warp computes eight tokens against 32 rows.
// The fragment coordinates follow PTX mma.m8n8k4.row.col. Four warps split K;
// their FP32 partials are reduced before any output rounding or fused epilogue.
__device__ __forceinline__ void volta_qpn_mma(float (&d)[8], unsigned a0, unsigned a1, unsigned b0,
                                              unsigned b1) {
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                 "{%0,%1,%2,%3,%4,%5,%6,%7};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
                   "+f"(d[6]), "+f"(d[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// E4M3's bias is eight below FP16. Preserve subnormals by rebiasing
// the packed half values with an exact power of two; retain both NaN codes.
__device__ __forceinline__ half2 volta_fp8_pair(unsigned lo, unsigned hi) {
    const unsigned packed = (lo & 255u) | ((hi & 255u) << 16);
    unsigned bits         = ((packed << 8) & 0x80008000u) | ((packed << 7) & 0x3f803f80u);
    if ((lo & 127) == 127) bits |= 0x7fffu;
    if ((hi & 127) == 127) bits |= 0x7fff0000u;
    return __hmul2(*reinterpret_cast<const half2*>(&bits), __float2half2_rn(256.f));
}

__device__ __forceinline__ std::int64_t volta_nvfp4_scale(int row, int group, int k) {
    return std::int64_t(row / 128) * (k / 64) * 512 + (group / 4) * 512 + (row & 31) * 16 +
           ((row % 128) / 32) * 4 + group % 4;
}

__device__ __forceinline__ float volta_weight(Fp8A16Operands p, int row, int k) {
    return common::volta::decode<common::volta::Format::E4m3>(p.codes[std::int64_t(row) * p.k + k]);
}

__device__ __forceinline__ float volta_weight(Nvfp4A16Operands p, int row, int k) {
    const auto code = p.codes[std::int64_t(row) * (p.k / 2) + k / 2];
    return common::volta::decode<common::volta::Format::E2m1>((code >> ((k & 1) * 4)) & 15) *
           common::volta::decode<common::volta::Format::E4m3>(
               p.scales[volta_nvfp4_scale(row, k / 16, p.k)]);
}

__device__ __forceinline__ float volta_weight_factor(Fp8A16Operands p, int row) {
    return __bfloat162float(p.scales[row]);
}

__device__ __forceinline__ float volta_weight_factor(Nvfp4A16Operands p, int) { return p.alpha; }

// The scale is a normal power of two with exponent in [-120, 120]. Its
// reciprocal is exact and normal too; avoid general FP32 division per input.
__device__ __forceinline__ float volta_inverse_scale(float scale) {
    return __uint_as_float((254u - (__float_as_uint(scale) >> 23)) << 23);
}

// Scale BF16 activations by an exact power of two to retain small values in
// FP16. Every conversion is checked; any lost value is retained in FP32.
// Neither path allocates or repacks persistent weights.
__device__ __forceinline__ float volta_activation_scale(const __nv_bfloat16* x, int count,
                                                        unsigned (&extrema)[8], bool& exact) {
    unsigned largest = 0, smallest = 0x7fff;
    // Supported contractions have K divisible by eight and aligned BF16 inputs.
    for (int i = threadIdx.x * 8; i < count; i += 128 * 8) {
        const uint4 packed = __ldg(reinterpret_cast<const uint4*>(x + i));
        const unsigned words[]{packed.x, packed.y, packed.z, packed.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned lo = words[j] & 0x7fff, hi = (words[j] >> 16) & 0x7fff;
            largest  = max(largest, max(lo, hi));
            smallest = min(smallest, min(lo ? lo : 0x7fff, hi ? hi : 0x7fff));
        }
    }
#pragma unroll
    for (int delta = 16; delta; delta /= 2) {
        largest  = max(largest, __shfl_down_sync(0xffffffffu, largest, delta));
        smallest = min(smallest, __shfl_down_sync(0xffffffffu, smallest, delta));
    }
    if ((threadIdx.x & 31) == 0) {
        extrema[threadIdx.x / 32]     = largest;
        extrema[4 + threadIdx.x / 32] = smallest;
    }
    __syncthreads();
    largest  = max(max(extrema[0], extrema[1]), max(extrema[2], extrema[3]));
    smallest = min(min(extrema[4], extrema[5]), min(extrema[6], extrema[7]));
    // A BF16 significand fits exactly in FP16 through exponent +15. At
    // exponent -17 its seven fraction bits still fit FP16 subnormals.
    const int exponent = min(120, max(-120, 15 - (int(largest >> 7) - 127)));
    exact = largest < 0x7f80 && smallest >= 128 && int(smallest >> 7) + exponent >= 110;
    return __uint_as_float(unsigned(exponent + 127) << 23);
}

// Keep rare values that cannot be represented by the scaled FP16 tile as
// explicit FP32 residuals: X = X_half / scale + residual. A bounded CTA-local
// list avoids penalizing a whole matrix for a few tiny SwiGLU outputs. Dense
// residuals and non-finite inputs use the full FP32 contraction instead.
struct VoltaActivationResiduals {
    static constexpr unsigned kCapacity = 64;
    unsigned count, nonfinite;
    int index[kCapacity];
    float value[kCapacity];
};

__device__ __forceinline__ bool volta_activation_residuals(const __nv_bfloat16* x, int count,
                                                           float scale,
                                                           VoltaActivationResiduals& residuals,
                                                           bool exact) {
    if (threadIdx.x == 0) {
        residuals.count     = 0;
        residuals.nonfinite = 0;
    }
    __syncthreads();
    // The exponent bounds prove exact conversion of every nonzero input.
    if (exact) return false;
    for (int i = threadIdx.x; i < count; i += 128) {
        const float original = __bfloat162float(x[i]);
        if (!isfinite(original)) {
            atomicOr(&residuals.nonfinite, 1u);
        } else {
            const float inverse     = volta_inverse_scale(scale);
            const float represented = __half2float(__float2half_rn(original * scale)) * inverse;
            const float correction  = original - represented;
            if (correction != 0.f) {
                const unsigned slot = atomicAdd(&residuals.count, 1u);
                if (slot < VoltaActivationResiduals::kCapacity) {
                    residuals.index[slot] = i;
                    residuals.value[slot] = correction;
                }
            }
        }
    }
    __syncthreads();
    const bool fallback =
        residuals.nonfinite || residuals.count > VoltaActivationResiduals::kCapacity;
    // Atomic insertion order is unspecified. Sort the small exceptional set so
    // graph replay and repeated state projections use the same reduction order.
    if (!fallback && threadIdx.x == 0) {
        for (unsigned i = 1; i < residuals.count; ++i) {
            const int index   = residuals.index[i];
            const float value = residuals.value[i];
            unsigned j        = i;
            while (j && residuals.index[j - 1] > index) {
                residuals.index[j] = residuals.index[j - 1];
                residuals.value[j] = residuals.value[j - 1];
                --j;
            }
            residuals.index[j] = index;
            residuals.value[j] = value;
        }
    }
    __syncthreads();
    return fallback;
}

template <class Operands>
__device__ __forceinline__ float volta_correct(Operands p, int row, int token, float value,
                                               const VoltaActivationResiduals& residuals) {
    const int begin = token * p.k;
    for (unsigned i = 0; i < residuals.count; ++i) {
        const int k = residuals.index[i] - begin;
        if (k >= 0 && k < p.k) value = fmaf(residuals.value[i], volta_weight(p, row, k), value);
    }
    return value;
}

template <bool Paired>
__device__ __forceinline__ int volta_qpn_row(int local, int rows) {
    if constexpr (Paired)
        return int(blockIdx.x) * 16 + local % 16 + (local / 16) * (rows / 2);
    else
        return int(blockIdx.x) * 32 + local;
}

template <int Tiles, bool Paired, class Operands, class Output, class Epilogue>
__global__ __launch_bounds__(128, Tiles == 1   ? 4
                                  : Tiles == 2 ? 4
                                               : 1) void volta_qpn_kernel(Operands p, Output output,
                                                                          Epilogue epilogue) {
    __shared__ float partial[4][Tiles * 8 * 32];
    __shared__ unsigned maxima[8];
    __shared__ VoltaActivationResiduals residuals;
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    const int qp     = (lane / 4) % 4;
    const int r      = lane % 4 + (lane / 16) * 4;
    const int token0 = int(blockIdx.y) * (Tiles * 8);
    const int tokens = min(Tiles * 8, p.tokens - token0);
    const auto* x    = p.x + std::int64_t(token0) * p.k;
    bool exact;
    const float activation_scale = volta_activation_scale(x, tokens * p.k, maxima, exact);
    const bool fallback =
        volta_activation_residuals(x, tokens * p.k, activation_scale, residuals, exact);
    if (fallback) {
        for (int e = threadIdx.x; e < tokens * 32; e += 128) {
            const int row = volta_qpn_row<Paired>(e % 32, p.rows);
            float value   = 0;
            if (row < p.rows) {
                for (int k = 0; k < p.k; ++k)
                    value = fmaf(__bfloat162float(x[std::int64_t(e / 32) * p.k + k]),
                                 volta_weight(p, row, k), value);
                value *= volta_weight_factor(p, row);
            }
            partial[0][e] = value;
        }
    } else {
        const int weight_row = volta_qpn_row<Paired>(qp * 8 + r, p.rows);
        const int safe_row   = min(weight_row, p.rows - 1);
        float c[Tiles][8]    = {};
        const int groups     = p.k / 16;
        const int begin      = warp * (groups / 4);
        const int end        = warp == 3 ? groups : begin + groups / 4;

#pragma unroll 4
        for (int group = begin; group < end; ++group) {
            half2 b[8];
            if constexpr (std::is_same_v<Operands, Nvfp4A16Operands>) {
                const uint2 packed = __ldg(reinterpret_cast<const uint2*>(
                    p.codes + std::int64_t(safe_row) * (p.k / 2) + group * 8));
                const float scale  = common::volta::decode<common::volta::Format::E4m3>(
                    p.scales[volta_nvfp4_scale(safe_row, group, p.k)]);
                // Decode with a separate rebias: scale * 16384 can overflow even
                // though each represented weight (at most 6 * 448) is finite.
                const half2 rebias = __float2half2_rn(16384.f);
                const half2 sc     = __float2half2_rn(scale);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const unsigned word = j < 4 ? packed.x : packed.y;
                    const int shift     = j % 4;
                    const unsigned bits = ((word << (12 - 4 * shift)) & 0x80008000u) |
                                          (shift < 3 ? (word << (9 - 4 * shift)) & 0x0e000e00u
                                                     : (word >> 3) & 0x0e000e00u);
                    const half2 raw = *reinterpret_cast<const half2*>(&bits);
                    b[j]            = __hmul2(__hmul2(raw, rebias), sc);
                }
            } else {
                const uint4 packed      = __ldg(reinterpret_cast<const uint4*>(
                    p.codes + std::int64_t(safe_row) * p.k + group * 16));
                const unsigned words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int first = (j / 4) * 8 + j % 4;
                    b[j]            = volta_fp8_pair(words[first / 4] >> ((first % 4) * 8),
                                                     words[(first + 4) / 4] >> ((first % 4) * 8));
                }
            }
#pragma unroll
            for (int tile = 0; tile < Tiles; ++tile) {
                half2 a[8];
                const int token = tile * 8 + r;
                uint4 raw[2]    = {};
                if (token < tokens) {
                    const auto* source = x + std::int64_t(token) * p.k + group * 16;
                    raw[0]             = *reinterpret_cast<const uint4*>(source);
                    raw[1]             = *reinterpret_cast<const uint4*>(source + 8);
                }
                const auto* values = reinterpret_cast<const __nv_bfloat16*>(raw);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int first = (j / 4) * 8 + j % 4;
                    a[j] =
                        __floats2half2_rn(__bfloat162float(values[first]) * activation_scale,
                                          __bfloat162float(values[first + 4]) * activation_scale);
                }
                const auto* av = reinterpret_cast<const unsigned*>(a);
                const auto* bv = reinterpret_cast<const unsigned*>(b);
#pragma unroll
                for (int j = 0; j < 8; j += 2)
                    volta_qpn_mma(c[tile], av[j], av[j + 1], bv[j], bv[j + 1]);
            }
        }
#pragma unroll
        for (int tile = 0; tile < Tiles; ++tile)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int token = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
                const int col   = (i & 1) | (lane & 2) | ((i >> 2) << 2);
                partial[warp][(tile * 8 + token) * 32 + qp * 8 + col] = c[tile][i];
            }
        __syncthreads();
        for (int e = threadIdx.x; e < tokens * 32; e += 128) {
            float value = 0;
#pragma unroll
            for (int w = 0; w < 4; ++w) value += partial[w][e];
            const int row = volta_qpn_row<Paired>(e % 32, p.rows);
            partial[0][e] =
                row < p.rows
                    ? volta_correct(p, row, e / 32, value * volta_inverse_scale(activation_scale),
                                    residuals) *
                          volta_weight_factor(p, row)
                    : 0;
        }
    }
    __syncthreads();
    for (int e = threadIdx.x; e < tokens * 32; e += 128) {
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

template <int Tiles, bool Paired = false, class Operands, class Output, class Epilogue>
void launch_volta_qpn(Operands p, Output output, Epilogue epilogue, cudaStream_t stream) {
    const dim3 grid((p.rows + 31) / 32, (p.tokens + Tiles * 8 - 1) / (Tiles * 8));
    volta_qpn_kernel<Tiles, Paired><<<grid, 128, 0, stream>>>(p, output, epilogue);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail
