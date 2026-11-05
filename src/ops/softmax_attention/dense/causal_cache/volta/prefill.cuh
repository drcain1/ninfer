#pragma once
// Native Volta MMA fragment math adapted from ninfer-all; see third_party/volta/NOTICE.
#include "core/device.h"
#include "ops/kv_cache/kernels/hadamard.cuh"
#include "ops/softmax_attention/kernels/causal/operands.h"
#include <cmath>
#include <cstdint>

// One warp owns eight grouped query rows and all 256 output dimensions. A CTA
// shares 32 gathered K/V rows across its warps. The full query batch is scheduled
// together to avoid partially occupied GPU waves at every scratch block boundary.
namespace ninfer::ops::detail::volta_prefill {
constexpr int kCausalHeadDim = 256;
constexpr int kQStride = 264;
constexpr int kStride = kQStride;
constexpr int kPStride = 40;
__device__ __forceinline__ float exp2_approx(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

template <bool RowB>
__device__ __forceinline__ void hmma(float (&c)[8], unsigned a0, unsigned a1, unsigned b0,
                                   unsigned b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    if constexpr (RowB) {
        asm volatile("mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 "
                     "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]),
                       "+f"(c[6]), "+f"(c[7])
                     : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
        return;
    } else {
        asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                     "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]),
                       "+f"(c[6]), "+f"(c[7])
                     : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
        return;
    }
#endif
    (void)c;
    (void)a0;
    (void)a1;
    (void)b0;
    (void)b1;
}
struct RowState {
    float m_a;
    float m_b;
    float l_a;
    float l_b;
    float acc[8][8];

    __device__ __forceinline__ RowState() : l_a(0.0f), l_b(0.0f) {
        m_a = -INFINITY;
        m_b = -INFINITY;
#pragma unroll
        for (int n = 0; n < 8; ++n) {
#pragma unroll
            for (int i = 0; i < 8; ++i) { acc[n][i] = 0.0f; }
        }
    }
};
struct SharedRows {
    const __half* k_s;
    const __half* v_s;
    int k0;

    __device__ __forceinline__ const __half* key(int k) const { return k_s + (k - k0) * kStride; }

    __device__ __forceinline__ uint4 key8(const __half* row, int d) const {
        return *reinterpret_cast<const uint4*>(row + d);
    }

    __device__ __forceinline__ const __half* value(int k, int dim0) const {
        return v_s + (k - k0) * kStride + dim0;
    }

    __device__ __forceinline__ uint2 value4(const __half* row, int n) const {
        return *reinterpret_cast<const uint2*>(row + 32 * n);
    }
};
__device__ __forceinline__ void attend_tile(const SharedRows& rows, const __half* q_row, __half* p_tile,
                                          int k0, int key_lo, int key_hi, int qabs_a, int qabs_b,
                                          float scale, RowState& st) {
    constexpr int D             = kCausalHeadDim;
    constexpr int Bc            = 32;
    constexpr int PStride       = kPStride;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int qp                = (lane >> 2) & 3;
    const int h                 = lane >> 4;
    const int a_row             = (lane & 3) + 4 * h;

    // S = Q K^T: this lane supplies the K row of key k0 + 8 qp + (lane % 4) + 4h.
    float s[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    {
        const int key    = k0 + qp * 8 + (lane & 3) + 4 * h;
        const bool live  = key >= key_lo && key < key_hi;
        const auto k_row = rows.key(live ? key : k0);
#pragma unroll 8
        for (int d = 0; d < D; d += 8) {
            const uint4 k8 = live ? rows.key8(k_row, d) : make_uint4(0, 0, 0, 0);
            const uint4 q8 = *reinterpret_cast<const uint4*>(q_row + d);
            hmma<false>(s, q8.x, q8.y, k8.x, k8.y);
            hmma<false>(s, q8.z, q8.w, k8.z, k8.w);
        }
    }
    // Masked, scaled scores and the tile's row maxima (rows a: i & 2 == 0, b: i & 2 == 2).
    float bm_a = -INFINITY, bm_b = -INFINITY;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int key    = k0 + qp * 8 + (i & 4) + (lane & 2) + (i & 1);
        const bool upper = (i & 2) != 0;
        const int qabs   = upper ? qabs_b : qabs_a;
        const bool keep  = qabs >= 0 && key >= key_lo && key < key_hi && key <= qabs;
        s[i]             = keep ? s[i] * scale : -INFINITY;
        if (upper) {
            bm_b = fmaxf(bm_b, s[i]);
        } else {
            bm_a = fmaxf(bm_a, s[i]);
        }
    }
#pragma unroll
    for (int mask = 2; mask <= 8; mask <<= 1) {
        bm_a = fmaxf(bm_a, __shfl_xor_sync(FullMask, bm_a, mask));
        bm_b = fmaxf(bm_b, __shfl_xor_sync(FullMask, bm_b, mask));
    }
    const float nm_a    = fmaxf(st.m_a, bm_a);
    const float nm_b    = fmaxf(st.m_b, bm_b);
    const float alpha_a = st.m_a == -INFINITY ? 0.0f : exp2_approx((st.m_a - nm_a) * Log2E);
    const float alpha_b = st.m_b == -INFINITY ? 0.0f : exp2_approx((st.m_b - nm_b) * Log2E);
    float bl_a = 0.0f, bl_b = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const bool upper = (i & 2) != 0;
        const float nm   = upper ? nm_b : nm_a;
        const float p =
            (nm > -INFINITY && s[i] > -INFINITY) ? exp2_approx((s[i] - nm) * Log2E) : 0.0f;
        if (upper) {
            bl_b += p;
        } else {
            bl_a += p;
        }
        const int row               = (lane & 1) + (i & 2) + 4 * h;
        const int col               = qp * 8 + (i & 4) + (lane & 2) + (i & 1);
        p_tile[row * PStride + col] = __float2half_rn(p);
    }
#pragma unroll
    for (int mask = 2; mask <= 8; mask <<= 1) {
        bl_a += __shfl_xor_sync(FullMask, bl_a, mask);
        bl_b += __shfl_xor_sync(FullMask, bl_b, mask);
    }
    st.l_a = st.l_a * alpha_a + bl_a;
    st.l_b = st.l_b * alpha_b + bl_b;
    st.m_a = nm_a;
    st.m_b = nm_b;
#pragma unroll
    for (int n = 0; n < 8; ++n) {
#pragma unroll
        for (int i = 0; i < 8; ++i) { st.acc[n][i] *= (i & 2) ? alpha_b : alpha_a; }
    }
    __syncwarp();
    // O += P V: four keys per product; quad pairs take eight of each 32 dimensions.
    const int value_dim0 = 8 * qp + 4 * h;
#pragma unroll 2
    for (int step = 0; step < Bc / 4; ++step) {
        const uint2 p4   = *reinterpret_cast<const uint2*>(p_tile + a_row * PStride + 4 * step);
        const int key    = k0 + 4 * step + (lane & 3);
        const bool live  = key >= key_lo && key < key_hi;
        const auto v_row = rows.value(live ? key : k0, value_dim0);
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            const uint2 v4 = live ? rows.value4(v_row, n) : make_uint2(0, 0);
            hmma<true>(st.acc[n], p4.x, p4.y, v4.x, v4.y);
        }
    }
    __syncwarp();
}

constexpr int kWarps = 12;
constexpr int kRows = kWarps * 8;
constexpr int kThreads = kWarps * 32;
constexpr int kSharedBytes = (kRows * kQStride + 64 * kQStride + kRows * kPStride) * 2;
static_assert(kSharedBytes <= 96 * 1024);

__global__ __launch_bounds__(kThreads) void prefill_kernel(
    CausalAttentionOperands p, const half* keys, const half* values,
    const int* valid_columns, int* exceptional) {
    const auto* positions = p.positions;
    const int tokens = p.width;
    const float scale = p.scale;
    extern __shared__ __align__(16) unsigned char shared[];
    half* qs = reinterpret_cast<half*>(shared);
    half* ks = qs + kRows * kQStride;
    half* vs = ks + 32 * kQStride;
    half* ps = vs + 32 * kQStride;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int group = 6, heads = 24, kvheads = 4;
    const int head = blockIdx.y;
    // Schedule the longest causal rows first; positions still determine every mask.
    const int row0 = (gridDim.x - 1 - blockIdx.x) * kRows;
    const int valid = valid_columns ? valid_columns[0] : p.width;
    const int rows = tokens * group;
    const auto q_index = [&](int row, int d) {
        return (std::int64_t(row / group) * heads + head * group + row % group) * 256 + d;
    };
    for (int r = warp; r < kRows; r += kWarps) {
        const int row = row0 + r;
        const bool live = row < rows && row / group < valid;
        float q[8];
#pragma unroll
        for (int i = 0; i < 8; ++i)
            q[i] = live ? __bfloat162float(p.q[q_index(row, lane + 32 * i)]) : 0.f;
        kv::normalized_hadamard_d256_inplace(q, lane);
        bool unsafe = false;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            unsafe |= !isfinite(q[i]) || fabsf(q[i]) > 65504.f;
            qs[r * kQStride + lane + 32 * i] = __float2half_rn(q[i]);
        }
        if (__any_sync(0xffffffffu, unsafe) && lane == 0) atomicExch(exceptional, 1);
    }
    int window = 0;
    for (int t = row0 / group; t <= min(row0 + kRows - 1, rows - 1) / group; ++t)
        if (t < valid) window = max(window, positions[t] + 1);
    const int lane_row = (lane & 1) + 4 * (lane >> 4);
    const int row_a = row0 + warp * 8 + lane_row, row_b = row_a + 2;
    const int pa = row_a < rows && row_a / group < valid
                       ? positions[row_a / group] : -1;
    const int pb = row_b < rows && row_b / group < valid
                       ? positions[row_b / group] : -1;
    int warp_position = max(pa, pb);
#pragma unroll
    for (int step = 16; step; step >>= 1)
        warp_position = max(warp_position, __shfl_xor_sync(0xffffffffu, warp_position, step));
    const half* qr = qs + (warp * 8 + (lane & 3) + 4 * (lane >> 4)) * kQStride;
    half* pp = ps + warp * 8 * kPStride;
    RowState state;
    const int load_row = tid / 8, part = tid % 8;
    uint4 kr[4], vr[4];
    const auto read = [&](int k0) {
        const int k = k0 + load_row;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const auto offset = (std::int64_t(k) * kvheads + head) * 256 + part * 32 + i * 8;
            kr[i] = k < window ? *reinterpret_cast<const uint4*>(keys + offset)
                              : make_uint4(0, 0, 0, 0);
            vr[i] = k < window ? *reinterpret_cast<const uint4*>(values + offset)
                              : make_uint4(0, 0, 0, 0);
        }
    };
    // The first eight warps stage K/V. All twelve consume those rows after the
    // block barrier; registers prefetch the next tile while MMA uses this one.
    if (window && tid < 256) read(0);
    for (int k0 = 0; k0 < window; k0 += 32) {
        if (tid < 256) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                *reinterpret_cast<uint4*>(ks + load_row * kQStride + part * 32 + i * 8) = kr[i];
                *reinterpret_cast<uint4*>(vs + load_row * kQStride + part * 32 + i * 8) = vr[i];
            }
        }
        __syncthreads();
        if (k0 + 32 < window && tid < 256) read(k0 + 32);
        if (warp_position >= k0)
            attend_tile(SharedRows{ks, vs, k0}, qr, pp, k0, 0, window, pa, pb, scale, state);
        __syncthreads();
    }
    const float inv_a = state.l_a > 0.f ? 1.f / state.l_a : 0.f;
    const float inv_b = state.l_b > 0.f ? 1.f / state.l_b : 0.f;
    const int qp = (lane >> 2) & 3;
#pragma unroll
    for (int n = 0; n < 8; ++n) {
#pragma unroll
        for (int i = 0; i < 8; i += 2) {
            const int row = (i & 2) ? row_b : row_a;
            if (row >= rows) continue;
            const int d = 32 * n + 8 * qp + (i & 4) + (lane & 2);
            const float inv = (i & 2) ? inv_b : inv_a;
            p.out[q_index(row, d)] = __float2bfloat16_rn(state.acc[n][i] * inv);
            p.out[q_index(row, d + 1)] = __float2bfloat16_rn(state.acc[n][i + 1] * inv);
        }
    }
}
inline void launch(CausalAttentionOperands p, const half* keys, const half* values,
                   const int* valid_columns, int* exceptional, cudaStream_t stream) {
    static const cudaError_t status = cudaFuncSetAttribute(
        prefill_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes);
    CUDA_CHECK(status);
    prefill_kernel<<<dim3((p.width * 6 + kRows - 1) / kRows, 4), kThreads, kSharedBytes, stream>>>(
        p, keys, values, valid_columns, exceptional);
}
} // namespace ninfer::ops::detail::volta_prefill
