#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// Register layouts follow the PTX m16n8 MMA and m8n8 ldmatrix contracts:
// https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma
// Volta lacks these instruction shapes. Keep their represented inputs and FP32
// accumulators intact using warp exchanges; no allocation or implicit FP16 cast.
namespace ninfer::ops::common::volta {

template <int Matrix, bool Transpose>
__device__ __forceinline__ unsigned load_matrix(unsigned address) {
    const int lane = threadIdx.x & 31;
    __syncwarp();
    if constexpr (!Transpose) {
        const unsigned row = __shfl_sync(0xffffffffu, address, Matrix * 8 + lane / 4);
        unsigned result;
        const unsigned location = row + (lane % 4) * 4;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(result) : "r"(location));
        return result;
    } else {
        const unsigned row0 = __shfl_sync(0xffffffffu, address, Matrix * 8 + (lane % 4) * 2);
        const unsigned row1 = __shfl_sync(0xffffffffu, address, Matrix * 8 + (lane % 4) * 2 + 1);
        unsigned lo, hi;
        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(lo) : "r"(row0 + (lane / 4) * 2));
        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(hi) : "r"(row1 + (lane / 4) * 2));
        return lo | (hi << 16);
    }
}

enum class Format { Bf16, F16, F32, S8, E4m3, E2m1 };

template <Format F>
__device__ __forceinline__ float decode(unsigned code) {
    if constexpr (F == Format::Bf16) return __uint_as_float(code << 16);
    if constexpr (F == Format::F16) return __half2float(__ushort_as_half(code));
    if constexpr (F == Format::F32) return __uint_as_float(code);
    if constexpr (F == Format::S8) return static_cast<float>(static_cast<signed char>(code));
    if constexpr (F == Format::E4m3) {
        const unsigned magnitude = code & 127;
        const unsigned sign      = (code & 128) << 24;
        if (magnitude == 127) return __uint_as_float(sign | 0x7fffffffu);
        if (magnitude < 8) {
            const float value = static_cast<float>(magnitude) * 0x1p-9f;
            return __uint_as_float(sign | __float_as_uint(value));
        }
        return __uint_as_float(sign | ((magnitude + 960) << 20));
    }
    if constexpr (F == Format::E2m1) {
        const unsigned magnitude = code & 7;
        const unsigned bits      = magnitude < 2
                                       ? magnitude * 0x3f000000u
                                       : (((magnitude / 2) + 126) << 23) | ((magnitude & 1) << 22);
        return __uint_as_float(((code & 8) << 28) | bits);
    }
}

__device__ __forceinline__ float exponent_scale(unsigned value) {
    return __uint_as_float(value ? (value & 255) << 23 : 0x00400000u);
}

template <Format F, int Bits, bool Scaled = false>
__device__ __forceinline__ void mma(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                    unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1,
                                    unsigned sa = 0x7f, unsigned sb = 0x7f) {
    constexpr int per_register = 32 / Bits;
    constexpr int half_k       = 4 * per_register;
    constexpr unsigned mask    = 0xffffffffu >> (32 - Bits);
    const int lane             = threadIdx.x & 31;
    const int row              = lane / 4;
    const int col              = (lane % 4) * 2;
    unsigned as0 = 0, as1 = 0, bs0 = 0, bs1 = 0;
    if constexpr (Scaled) {
        as0 = __shfl_sync(0xffffffffu, sa, row * 4);
        as1 = __shfl_sync(0xffffffffu, sa, row * 4 + 1);
        bs0 = __shfl_sync(0xffffffffu, sb, col * 4);
        bs1 = __shfl_sync(0xffffffffu, sb, (col + 1) * 4);
    }
    // NVFP4 has one scale per K16 group. Decode it once per group, not
    // once per scalar multiply. Other formats share one scale across K.
    constexpr int group_k = F == Format::E2m1 ? 16 : 2 * half_k;
#pragma unroll 1
    for (int group = 0; group < 2 * half_k / group_k; ++group) {
        float sx0 = 1, sx1 = 1, sy0 = 1, sy1 = 1;
        if constexpr (Scaled && F == Format::E2m1) {
            const int offset = group * 8;
            sx0              = decode<Format::E4m3>((as0 >> offset) & 255);
            sx1              = decode<Format::E4m3>((as1 >> offset) & 255);
            sy0              = decode<Format::E4m3>((bs0 >> offset) & 255);
            sy1              = decode<Format::E4m3>((bs1 >> offset) & 255);
        } else if constexpr (Scaled) {
            sx0 = exponent_scale(as0 & 255);
            sx1 = exponent_scale(as1 & 255);
            sy0 = exponent_scale(bs0 & 255);
            sy1 = exponent_scale(bs1 & 255);
        }
#pragma unroll 1
        for (int item = 0; item < group_k; item += per_register) {
            const int k        = group * group_k + item;
            const int owner    = (k % half_k) / per_register;
            const unsigned px0 = __shfl_sync(0xffffffffu, k < half_k ? a0 : a2, row * 4 + owner);
            const unsigned px1 = __shfl_sync(0xffffffffu, k < half_k ? a1 : a3, row * 4 + owner);
            const unsigned py0 = __shfl_sync(0xffffffffu, k < half_k ? b0 : b1, col * 4 + owner);
            const unsigned py1 =
                __shfl_sync(0xffffffffu, k < half_k ? b0 : b1, (col + 1) * 4 + owner);
            // Exchange a packed word once, then consume all of its scalar lanes.
#pragma unroll
            for (int scalar = 0; scalar < per_register; ++scalar) {
                const int shift = scalar * Bits;
                const float x0  = sx0 * decode<F>((px0 >> shift) & mask);
                const float x1  = sx1 * decode<F>((px1 >> shift) & mask);
                const float y0  = sy0 * decode<F>((py0 >> shift) & mask);
                const float y1  = sy1 * decode<F>((py1 >> shift) & mask);
                c0              = fmaf(x0, y0, c0);
                c1              = fmaf(x0, y1, c1);
                c2              = fmaf(x1, y0, c2);
                c3              = fmaf(x1, y1, c3);
            }
        }
    }
}

// Integer MMA retains an integer accumulator, including wraparound at 32 bits.
__device__ __forceinline__ void mma_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0, unsigned a1,
                                       unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    const int lane = threadIdx.x & 31, row = lane / 4, col = (lane % 4) * 2;
#pragma unroll
    for (int k = 0; k < 32; k += 4) {
        const int owner = (k % 16) / 4;
        const int x0    = __shfl_sync(0xffffffffu, k < 16 ? a0 : a2, row * 4 + owner);
        const int x1    = __shfl_sync(0xffffffffu, k < 16 ? a1 : a3, row * 4 + owner);
        const int y0    = __shfl_sync(0xffffffffu, k < 16 ? b0 : b1, col * 4 + owner);
        const int y1    = __shfl_sync(0xffffffffu, k < 16 ? b0 : b1, (col + 1) * 4 + owner);
        // Volta supports signed four-byte dot products with an integer accumulator.
        c0 = __dp4a(x0, y0, c0);
        c1 = __dp4a(x0, y1, c1);
        c2 = __dp4a(x1, y0, c2);
        c3 = __dp4a(x1, y1, c3);
    }
}
} // namespace ninfer::ops::common::volta
