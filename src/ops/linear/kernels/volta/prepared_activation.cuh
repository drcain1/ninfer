#pragma once
#include "ops/linear/kernels/volta/packed.cuh"

namespace ninfer::ops::detail {
// Prepared once per token tile in caller-owned workspace. The exceptional
// correction list preserves the original BF16 input range without host reads.
struct VoltaActivationTile {
    float scale;
    bool fallback;
    VoltaActivationResiduals residuals;
};

struct VoltaPreparedActivation {
    half* values               = nullptr;
    VoltaActivationTile* tiles = nullptr;
};

template <int Tokens>
__global__ void volta_prepare_activation(const __nv_bfloat16* x, int k, int tokens,
                                         VoltaPreparedActivation prepared) {
    __shared__ unsigned extrema[8];
    __shared__ VoltaActivationResiduals residuals;
    const int begin = int(blockIdx.x) * Tokens * k;
    const int count = min(Tokens, tokens - int(blockIdx.x) * Tokens) * k;
    bool exact;
    const float scale   = volta_activation_scale(x + begin, count, extrema, exact);
    const bool fallback = volta_activation_residuals(x + begin, count, scale, residuals, exact);
    auto& tile          = prepared.tiles[blockIdx.x];
    if (threadIdx.x == 0) {
        tile.scale               = scale;
        tile.fallback            = fallback;
        tile.residuals.count     = residuals.count;
        tile.residuals.nonfinite = residuals.nonfinite;
    }
    if (!fallback && threadIdx.x < residuals.count) {
        tile.residuals.index[threadIdx.x] = residuals.index[threadIdx.x];
        tile.residuals.value[threadIdx.x] = residuals.value[threadIdx.x];
    }
    for (int i = threadIdx.x * 8; i < count; i += 128 * 8) {
        const uint4 raw    = __ldg(reinterpret_cast<const uint4*>(x + begin + i));
        const auto* values = reinterpret_cast<const __nv_bfloat16*>(&raw);
        __align__(16) half result[8];
#pragma unroll
        for (int j = 0; j < 8; ++j)
            result[j] = __float2half_rn(__bfloat162float(values[j]) * scale);
        *reinterpret_cast<uint4*>(prepared.values + begin + i) =
            *reinterpret_cast<const uint4*>(result);
    }
}

} // namespace ninfer::ops::detail
