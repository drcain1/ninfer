#pragma once

#include "ops/common/host/launch.h"
#include "ops/common/numeric/math.h"
#include "ops/linear/kernels/bf16/bf16_operands.h"
#include "ops/linear/kernels/bf16/bf16_a16_mma.cuh"

namespace ninfer::ops::detail {

template <class Schedule, int Splits, class Output, class Epilogue>
void launch_bf16_mma_partitions(const Bf16A16Operands& p, Output output, Epilogue epilogue,
                                cudaStream_t stream) {
    static_assert(Splits >= 1 && Splits <= 32);
    static_assert(Schedule::kThreads == Schedule::kWarpsRows * Schedule::kWarpsTokens * 32,
                  "cp.async MMA schedules must not reserve TMA producer warps");
    validate_bf16_operands<Schedule>(p);
    if ((!bf16_predicated_rows<Schedule> && p.rows % Schedule::kBlockRows) ||
        p.k % ((bf16_predicated_k<Schedule> ? 8 : Schedule::kBlockK) * Splits))
        throw std::invalid_argument("BF16 MMA requires compatible row/K tiles in every split");
    common::for_each_token_slice(p.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const auto blocks =
            static_cast<std::int64_t>(bf16_predicated_rows<Schedule>
                                          ? common::div_up(p.rows, Schedule::kBlockRows)
                                          : p.rows / Schedule::kBlockRows) *
            common::div_up(count, Schedule::kBlockTokens);
        if (blocks > 2147483647LL)
            throw std::invalid_argument("BF16 MMA grid exceeds CUDA grid.x capacity");
        const dim3 grid(static_cast<unsigned>(blocks), 1, Splits);
        const auto launch = [&]<bool Full>() {
            constexpr auto kernel = bf16_a16_mma_kernel<Schedule, Full, Output, Epilogue, Splits>;
            const int bytes =
                common::prepare_shared<linear_mma_shared_bytes<Schedule, Epilogue>, kernel, true>();
            kernel<<<grid, Schedule::kThreads, bytes, stream>>>(p.x, p.weight, output, epilogue,
                                                                p.rows, p.k, offset, count);
            CUDA_CHECK(cudaGetLastError());
        };
        if (count % Schedule::kBlockTokens == 0)
            launch.template operator()<true>();
        else
            launch.template operator()<false>();
    });
}

} // namespace ninfer::ops::detail
