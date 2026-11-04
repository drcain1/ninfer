#pragma once
#include "ops/common/cuda/volta_mma.cuh"

namespace ninfer::ops::detail::gated_delta_net::v100 {
// GDN repeatedly contracts tiny fixed K8/K16 tiles. Unrolling removes loop and
// operand-selection overhead while preserving represented inputs, FP32 FMAs,
// and their accumulation order. No FP16 conversion of the recurrent state.
template <common::volta::Format Format, int Bits>
__device__ __forceinline__ void contract(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                         unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
    constexpr int Words = 32 / Bits, HalfK = 4 * Words;
    constexpr unsigned Mask = 0xffffffffu >> (32 - Bits);
    const int lane = threadIdx.x & 31, row = lane / 4, col = (lane % 4) * 2;
#pragma unroll
    for (int k = 0; k < 2 * HalfK; k += Words) {
        const int owner   = (k % HalfK) / Words;
        const unsigned x0 = __shfl_sync(0xffffffffu, k < HalfK ? a0 : a2, row * 4 + owner);
        const unsigned x1 = __shfl_sync(0xffffffffu, k < HalfK ? a1 : a3, row * 4 + owner);
        const unsigned y0 = __shfl_sync(0xffffffffu, k < HalfK ? b0 : b1, col * 4 + owner);
        const unsigned y1 = __shfl_sync(0xffffffffu, k < HalfK ? b0 : b1, (col + 1) * 4 + owner);
#pragma unroll
        for (int word = 0; word < Words; ++word) {
            const float ax = common::volta::decode<Format>((x0 >> (word * Bits)) & Mask);
            const float ay = common::volta::decode<Format>((x1 >> (word * Bits)) & Mask);
            const float bx = common::volta::decode<Format>((y0 >> (word * Bits)) & Mask);
            const float by = common::volta::decode<Format>((y1 >> (word * Bits)) & Mask);
            c0             = fmaf(ax, bx, c0);
            c1             = fmaf(ax, by, c1);
            c2             = fmaf(ay, bx, c2);
            c3             = fmaf(ay, by, c3);
        }
    }
}

__device__ __forceinline__ void mma_bf16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                         unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
    contract<common::volta::Format::Bf16, 16>(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
}

__device__ __forceinline__ void mma_tf32_bits(float& c0, float& c1, float& c2, float& c3,
                                              unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                              unsigned b0, unsigned b1) {
    contract<common::volta::Format::F32, 32>(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
}
} // namespace ninfer::ops::detail::gated_delta_net::v100
