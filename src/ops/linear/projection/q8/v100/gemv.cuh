#pragma once

#include "ops/linear/kernels/q8/q8_gemv_launch.cuh"

namespace ninfer::ops::detail {

// Small Q8 projections use FP32 SIMT contractions on Volta. These avoid
// emulating newer MMA instructions and preserve the full BF16 input range.
template <int Tokens>
void launch_q8_volta_gemv(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const auto operands = q8_linear_operands(x, weight);
    const LinearBf16Output output{static_cast<__nv_bfloat16*>(out.data), weight.n};
    if constexpr (Tokens == 1)
        launch_q8_a16_gemv<Q8A16GemvSchedule<4, 1, 2>>(operands, output, LinearIdentityEpilogue{},
                                                       stream);
    else
        launch_q8_a16_gemv<Q8A16GemvTileSchedule<Tokens, 4, 1, 2, 0>>(
            operands, output, LinearIdentityEpilogue{}, stream);
}

} // namespace ninfer::ops::detail
