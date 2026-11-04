#pragma once
#include "ops/quantized_weight.h"

namespace ninfer::test {
// Test-owned byte permutation. Numerical references must be decoded before this
// transform, so they never share the production kernel's fragment addressing.
inline void pack_volta_weight(quantized_weight::PackedWeight& weight) {
    auto original = weight.payload;
    const int n = weight.weight.n, k = weight.weight.k;
    const bool nv   = weight.weight.qtype == QType::NVFP4;
    const auto base = weight.scale_plane_offset;
    for (int row = 0; row < n; ++row)
        for (int group = 0; group < k / 16; ++group) {
            const int lane      = row % 32 / 8 * 4 + row % 4 + row % 8 / 4 * 16;
            const auto fragment = (std::size_t(row / 32) * (k / 16) + group) * 32 + lane;
            if (!nv) {
                std::copy_n(original.data() + std::size_t(row) * k + group * 16, 16,
                            weight.payload.data() + fragment * 16);
                continue;
            }
            constexpr int order[]{0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15};
            for (int j = 0; j < 8; ++j) {
                const auto code = [&](int at) {
                    return (original[std::size_t(row) * (k / 2) + group * 8 + at / 2] >>
                            ((at % 2) * 4)) &
                           15;
                };
                weight.payload[fragment * 8 + j] =
                    code(order[j * 2]) | (code(order[j * 2 + 1]) << 4);
            }
            const auto source = (std::size_t(row / 128) * (k / 64) + group / 4) * 512 +
                                row % 32 * 16 + (row % 128) / 32 * 4 + group % 4;
            weight.payload[base + fragment] = original[base + source];
        }
    weight.weight.layout = QuantLayout::VoltaQpn;
}
} // namespace ninfer::test
