#include "core/platform.h"
#include "ops/linear/projection/q4/q4_shapes.h"

namespace ninfer::ops::detail {
#if NINFER_TARGET_V100
namespace {
void launch_columns(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    for (int column = 0; column < x.ne[1]; ++column) {
        const Tensor input = x.slice(1, column, 1);
        Tensor output      = out.slice(1, column, 1);
        launch_q4_a16_gemv_r4_w1_direct(input, weight, output, stream);
    }
}
} // namespace
#endif

Q4Launch select_q4_n131072_k5120(std::int32_t tokens) {
    if (tokens <= 1) return launch_q4_a16_gemv_r4_w1_direct;
#if NINFER_TARGET_V100
    // Small proposal batches avoid the emulated newer-architecture MMA path.
    if (tokens <= 4) return launch_columns;
#endif
    // Remaining schedules were selected on RTX 5090.
    if (tokens <= 4) return launch_q4_a16_sliced_k5120_t4;
    if (tokens <= 8) return launch_q4_a16_sliced_r32_t8_w4_s2;
    if (tokens <= 16) return launch_q4_a16_sliced_r32_t16_w4_s1;
    if (tokens <= 32) return launch_q4_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 64) return launch_q4_a16_mma_r64_t64_k128_s2_a1;
    if (tokens <= 80) return launch_q4_a16_mma_r64_t80;
    if (tokens <= 96) return launch_q4_a16_mma_r64_t96;
    if (tokens <= 112) return launch_q4_a16_mma_r64_t112;
    return launch_q4_a16_mma_r64_t128;
}
} // namespace ninfer::ops::detail
