#include "core/platform.h"
#if NINFER_TARGET_V100
#    include "ops/linear/kernels/volta/prepared.h"
#endif
#include "ops/linear/add/nvfp4/dispatch.h"

namespace ninfer::ops::detail {
namespace {
Nvfp4LinearAddPlan select_plan(std::int32_t rows, std::int32_t k, LinearPolicy policy,
                               std::int32_t tokens) {
    if (tokens > 0 && rows == 5120) {
        if (k == 6144) return select_nvfp4_add_n5120_k6144(policy, tokens);
        if (k == 17408) return select_nvfp4_add_n5120_k17408(policy, tokens);
    }
    throw std::invalid_argument("nvfp4 linear_add: unsupported shape or token extent");
}
} // namespace

std::size_t nvfp4_linear_add_workspace_capacity_bytes(std::int32_t rows, std::int32_t k,
                                                      LinearPolicy policy, std::int32_t first,
                                                      std::int32_t last) {
    if (first <= 0 || last < first)
        throw std::invalid_argument("nvfp4 linear_add: invalid token interval");
    (void)select_plan(rows, k, policy, first);
    const auto plan = select_plan(rows, k, policy, last);
#if NINFER_TARGET_V100
    return volta_linear_capacity(rows, k, last);
#endif
    return plan.a16 ? 0 : nvfp4_a4_workspace_capacity_bytes(last, k);
}

void nvfp4_linear_add_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                               MemoryArena& arena, cudaStream_t stream) {
    const auto plan = select_plan(w.n, w.k, policy, x.ne[1]);
#if NINFER_TARGET_V100
    auto volta_scope = arena.scope();
    return volta_linear_add(x, w, out, allocate_volta_linear(arena, w.n, w.k, x.ne[1]), stream);
#endif
    if (plan.a16) return plan.a16(x, w, out, stream);
    auto scope         = arena.scope();
    const auto scratch = allocate_nvfp4_a4_workspace(arena, x.ne[1], w.k);
    launch_nvfp4_a4_quantize(x, w, scratch, plan.scale_layout, stream);
    plan.quantized(x, w, out, scratch, stream);
}
} // namespace ninfer::ops::detail
