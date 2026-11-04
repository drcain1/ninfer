#pragma once

#include "ops/common/host/launch.h"
#include "ops/common/numeric/math.h"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/kernels/fp8/fp8_a8_mma.cuh"

namespace ninfer::ops::detail {
template <class Schedule, class Output, class Epilogue, class Rows = Fp8IdentityRows>
void launch_fp8_a8_mma(const Fp8A8Operands& p, Output output, Epilogue epilogue,
                       cudaStream_t stream, Rows rows = {}) {
    validate_fp8_operands<Schedule>(p);
    if (p.rows % Schedule::kBlockRows || p.k % Schedule::kBlockK ||
        p.k / Schedule::kBlockK < Schedule::kStages)
        throw std::invalid_argument("FP8 A8 MMA requires complete row/K tiles and enough K stages");
    common::for_each_token_slice(p.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const int blocks =
            p.rows / Schedule::kBlockRows * common::div_up(count, Schedule::kBlockTokens);
        const auto launch = [&]<bool Full>() {
            constexpr auto kernel = fp8_a8_mma_kernel<Schedule, Full, Epilogue, Output, Rows>;
            const int bytes =
                common::prepare_shared<linear_mma_shared_bytes<Schedule, Epilogue>, kernel>();
            kernel<<<blocks, Schedule::kThreads, bytes, stream>>>(p, output, epilogue, rows, offset,
                                                                  count);
            CUDA_CHECK(cudaGetLastError());
        };
        if (count % Schedule::kBlockTokens == 0)
            launch.template operator()<true>();
        else
            launch.template operator()<false>();
    });
}
} // namespace ninfer::ops::detail
