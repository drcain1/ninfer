#include "core/platform.h"
#if NINFER_TARGET_V100
#    include "ops/linear/kernels/volta/prepared.h"
#endif
#include "ops/linear/split/fp8/shapes/n14336_k5120/launch.h"

namespace ninfer::ops::detail::linear_split::fp8_n14336_k5120 {
namespace {
bool quantized(LinearPolicy policy, int columns) {
    return !NINFER_TARGET_V100 && allows_a8(policy) && columns >= 17;
}

std::size_t capacity(LinearPolicy policy, std::int32_t, std::int32_t last) {
#if NINFER_TARGET_V100
    return volta_linear_capacity(14336, 5120, last);
#endif
    return quantized(policy, last)
               ? fp8_a8_workspace_capacity_bytes(last, 5120, partial_capacity(last))
               : 0;
}

void launch(const Tensor& x, Weights weights, std::span<const Tensor> outputs,
            MemoryArena* workspace, cudaStream_t stream) {
#if NINFER_TARGET_V100
    if (workspace) {
        auto scope = workspace->scope();
        return volta_linear_projection(
            x, weights.first, outputs,
            allocate_volta_linear(*workspace, weights.first.n, weights.first.k, x.ne[1]), stream);
    }
    if (weights.first.layout == QuantLayout::VoltaQpn)
        throw std::invalid_argument("Volta-packed split requires caller workspace");
    return launch_a16(x, weights.first, outputs, stream);
#endif
    if (!quantized(weights.policy, x.ne[1])) {
        launch_a16(x, weights.first, outputs, stream);
        return;
    }
    const auto scratch =
        allocate_fp8_a8_workspace(*workspace, x.ne[1], 5120, partial_capacity(x.ne[1]));
    launch_a8(x, weights.first, outputs, scratch, stream);
}
} // namespace
} // namespace ninfer::ops::detail::linear_split::fp8_n14336_k5120

namespace ninfer::ops::detail::linear_split {
const SplitShape kFp8SplitN14336K5120{14336, 5120, kSplitRows<6144, 1024, 6144, 1024>,
                                      fp8_n14336_k5120::launch, fp8_n14336_k5120::capacity};
} // namespace ninfer::ops::detail::linear_split
