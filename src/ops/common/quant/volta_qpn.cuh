#pragma once
#include <cstdint>

namespace ninfer::ops::common {
// Physical 16-value fragment for the lossless volta_qpn_v1 weight layout.
__device__ __forceinline__ std::int64_t volta_qpn_fragment(int row, int group, int k) {
    const int lane = row % 32 / 8 * 4 + row % 4 + row % 8 / 4 * 16;
    return (std::int64_t(row / 32) * (k / 16) + group) * 32 + lane;
}
} // namespace ninfer::ops::common
