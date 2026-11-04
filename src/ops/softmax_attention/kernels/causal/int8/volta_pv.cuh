#pragma once

#include "ops/softmax_attention/kernels/causal/epilogue.cuh"
#include "ops/softmax_attention/kernels/causal/tile_io.cuh"

namespace ninfer::ops::detail {

// Keep the probability/value contraction in native m8n8k4 fragments throughout
// the online softmax loop. Four quadpairs cover a 16-query by 16-value tile;
// each lane owns eight FP32 accumulators. This avoids translating each K16
// product through the m16n8 compatibility layout and its warp exchanges.
// P and V retain the existing FP16 representation; only their contraction and
// accumulator layout change. The INT8 QK path and softmax statistics are shared.
template <int RowTiles, int KeyRows, int OutputTiles>
struct VoltaPvTile {
    static_assert(OutputTiles > 0);
    float values[OutputTiles][8]{};

    __device__ __forceinline__ void accumulate(const half* probabilities, const half* v,
                                                const float* alpha, int warp, int lane) {
        const int quadpair = (lane / 4) % 4;
        const int r        = (lane & 3) + ((lane & 16) >> 2);
        const int row_base = (warp % RowTiles) * 16;
        const int row      = row_base + (quadpair / 2) * 8 + (lane & 1) + ((lane & 16) >> 2);
        const auto* p      = probabilities + row_base * KeyRows;
        const int p_row    = (quadpair / 2) * 8 + r;
#pragma unroll
        for (int n = 0; n < OutputTiles; ++n) {
#pragma unroll
            for (int i = 0; i < 8; ++i) values[n][i] *= alpha[row + (i & 2)];
            const int v_col = (warp / RowTiles) * OutputTiles * 16 + n * 16 +
                              (quadpair % 2) * 8 + r;
#pragma unroll
            for (int k = 0; k < KeyRows; k += 4) {
                const unsigned a0 = *reinterpret_cast<const unsigned*>(
                    &p[p_row * KeyRows + causal_probability_swizzle<KeyRows>(p_row, k)]);
                const unsigned a1 = *reinterpret_cast<const unsigned*>(
                    &p[p_row * KeyRows + causal_probability_swizzle<KeyRows>(p_row, k + 2)]);
                const half2 b0 = __halves2half2(
                    v[k * 256 + common::swizzle_xor_64(k, v_col)],
                    v[(k + 1) * 256 + common::swizzle_xor_64(k + 1, v_col)]);
                const half2 b1 = __halves2half2(
                    v[(k + 2) * 256 + common::swizzle_xor_64(k + 2, v_col)],
                    v[(k + 3) * 256 + common::swizzle_xor_64(k + 3, v_col)]);
                auto& d = values[n];
                asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                             "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                             "{%0,%1,%2,%3,%4,%5,%6,%7};"
                             : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]),
                               "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                             : "r"(a0), "r"(a1), "r"(*reinterpret_cast<const unsigned*>(&b0)),
                               "r"(*reinterpret_cast<const unsigned*>(&b1)));
            }
        }
    }

    template <class Geometry>
    __device__ __forceinline__ void store(float* output, int warp, int lane, int kv_head,
                                          int tokens, int begin, int split, int width) const {
        const int quadpair = (lane / 4) % 4;
#pragma unroll
        for (int n = 0; n < OutputTiles; ++n) {
#pragma unroll
            for (int i = 0; i < 8; i += 2) {
                const int row = (warp % RowTiles) * 16 + (quadpair / 2) * 8 + (i & 2) +
                                (lane & 1) + ((lane & 16) >> 2);
                const int d = (warp / RowTiles) * OutputTiles * 16 + n * 16 +
                              (quadpair % 2) * 8 + (lane & 2) + (i / 4) * 4;
                if (row < tokens * Geometry::GroupSize) {
                    int q_head = 0, token = 0;
                    causal_row_to_qt<Geometry>(row, kv_head, q_head, token);
                    const auto dst =
                        causal_partial_index<Geometry>(q_head, d, begin + token, split, width);
                    causal_store_partial_pair(output + dst, values[n][i], values[n][i + 1]);
                }
            }
        }
    }
};

} // namespace ninfer::ops::detail
