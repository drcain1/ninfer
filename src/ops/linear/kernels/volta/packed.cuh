#pragma once
#include "ops/linear/kernels/volta/qpn.cuh"
#include "ops/common/quant/volta_qpn.cuh"

namespace ninfer::ops::detail {
// Fragment layout is a lossless permutation of the persisted numeric words.
struct VoltaPackedFp8 : Fp8A16Operands {};

struct VoltaPackedNvfp4 : Nvfp4A16Operands {};
template <class T>
inline constexpr bool volta_packed_v =
    std::is_same_v<T, VoltaPackedFp8> || std::is_same_v<T, VoltaPackedNvfp4>;
template <class T>
inline constexpr bool volta_nvfp4_v = std::is_base_of_v<Nvfp4A16Operands, T>;

using common::volta_qpn_fragment;

__device__ __forceinline__ float volta_weight(VoltaPackedFp8 p, int row, int k) {
    const auto code = p.codes[volta_qpn_fragment(row, k / 16, p.k) * 16 + k % 16];
    return common::volta::decode<common::volta::Format::E4m3>(code);
}

__device__ __forceinline__ float volta_weight(VoltaPackedNvfp4 p, int row, int k) {
    const auto group    = volta_qpn_fragment(row, k / 16, p.k);
    const int logical   = k % 8;
    const int nibble    = k % 16 / 8 * 8 + (logical % 2) * 4 + logical / 2;
    const unsigned code = (p.codes[group * 8 + nibble / 2] >> ((nibble % 2) * 4)) & 15;
    return common::volta::decode<common::volta::Format::E2m1>(code) *
           common::volta::decode<common::volta::Format::E4m3>(p.scales[group]);
}

// E4M3 words divided by 256 are exactly representable in FP16, including
// subnormals. Restore the power of two in FP32 before applying row scales.
__device__ __forceinline__ half2 volta_fp8_unscaled(unsigned lo, unsigned hi) {
    const unsigned packed = (lo & 255u) | ((hi & 255u) << 16);
    unsigned bits         = ((packed << 8) & 0x80008000u) | ((packed << 7) & 0x3f803f80u);
    if ((lo & 127) == 127) bits |= 0x7fffu;
    if ((hi & 127) == 127) bits |= 0x7fff0000u;
    return *reinterpret_cast<const half2*>(&bits);
}
} // namespace ninfer::ops::detail
