#include "ops/linear/projection/q8/v100/small.h"
#include "ops/linear/kernels/q8/q8_gemv_launch.cuh"

namespace ninfer::ops::detail {
void launch_q8_volta_eight_tokens(const Tensor& x, const Weight& w, Tensor& out,
                                  cudaStream_t stream) {
    launch_q8_a16_gemv<Q8A16GemvTileSchedule<8, 2, 4, 2, 0>>(
        q8_linear_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
        LinearIdentityEpilogue{}, stream);
}
} // namespace ninfer::ops::detail
