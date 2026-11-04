#include "core/platform.h"
#if NINFER_TARGET_V100
#    include "ops/linear/kernels/volta/prepared.h"
#endif
#pragma once

#include "ops/linear_attention/gated_delta_net/input_conv/fp8/launch.h"
#include "ops/linear_attention/gated_delta_net/input_conv/plan.h"
#include "ops/linear/kernels/fp8/fp8_a8_plan.h"
#include "ops/linear/kernels/nvfp4/nvfp4_a4_plan.h"

namespace ninfer::ops::detail::gdn_input {

struct Scratch {
    Tensor projected;
    Fp8A8Workspace fp8;
    Nvfp4A4Workspace nvfp4;
#if NINFER_TARGET_V100
    VoltaLinearScratch volta;
#endif
};

// The same allocation recipe runs on WorkspaceLayoutBuilder and on the call-local arena.
template <class Allocator>
Scratch allocate_workspace(Allocator& allocator, const WeightProfile& weights, ConvPlan plan,
                           std::int32_t columns, bool materialized_snapshot = false) {
    Scratch out;
    if (!plan.materialized()) return out;
    if (materialized_snapshot)
        out.projected = allocator.alloc(DType::BF16, {weights.geometry.channels(), columns}, 256);
    if (plan.projection.compute == Compute::A8)
        out.fp8 = allocate_fp8_a8_workspace(allocator, columns, weights.geometry.input_rows,
                                            fp8_gdn_input_partial_capacity_bytes(columns));
    if (plan.projection.compute == Compute::A4)
        out.nvfp4 = allocate_nvfp4_a4_workspace(allocator, columns, weights.geometry.input_rows);
#if NINFER_TARGET_V100
    if (weights.format == Format::Fp8 || weights.format == Format::Nvfp4)
        out.volta = allocate_volta_linear(allocator,
                                          weights.geometry.channels() + weights.geometry.value_rows,
                                          weights.geometry.input_rows, columns);
#endif
    return out;
}

inline std::size_t workspace_bytes(const WeightProfile& weights, ConvPlan plan,
                                   std::int32_t columns, bool materialized_snapshot = false) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_workspace(layout, weights, plan, columns, materialized_snapshot);
    return layout.peak_bytes(1);
}

} // namespace ninfer::ops::detail::gdn_input
