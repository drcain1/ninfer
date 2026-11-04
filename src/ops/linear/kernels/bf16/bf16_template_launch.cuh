#pragma once

#include "ops/common/host/launch.h"
#include "ops/common/numeric/math.h"
#include "ops/linear/common/epilogue.cuh"

#include "ops/linear/kernels/bf16/bf16_operands.h"
#include "ops/linear/kernels/bf16/bf16_a16_gemv.cuh"
#include "ops/linear/kernels/bf16/bf16_a16_simt.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_a16_sliced_k_mma.cuh"
#include "ops/linear/kernels/bf16/bf16_a16_tma_mma.cuh"

namespace ninfer::ops::detail {

template <class Schedule, class Output, class Epilogue>
void launch_bf16_a16_gemv(const Bf16A16Operands& p, Output output, Epilogue epilogue,
                          cudaStream_t stream) {
    validate_bf16_operands<Schedule>(p);
    if (p.tokens != 1 || p.rows % Schedule::kBlockRows ||
        p.k % (bf16_predicated_k<Schedule>
                   ? (Schedule::kValuesPerLane < 8 ? 8 : Schedule::kValuesPerLane)
                   : Schedule::kWarpsPerRow * 32 * Schedule::kValuesPerLane))
        throw std::invalid_argument("BF16 GEMV requires T=1 and complete row/K tiles");
    if constexpr (requires { Epilogue::kRowTokens; }) {
        static_assert(Epilogue::kRowTokens == 1, "BF16 GEMV row consumers require one token");
    }
    constexpr auto kernel = bf16_a16_gemv_kernel<Schedule, Output, Epilogue>;
    int bytes             = 0;
    if constexpr (Schedule::kActivationAccess == Bf16ActivationAccess::Shared) {
        constexpr int static_bytes = sizeof(Bf16GemvSharedStorage<Schedule>);
        constexpr int max_dynamic  = (99 * 1024 - static_bytes) / 128 * 128;
        if (p.k > max_dynamic / 2)
            throw std::invalid_argument("BF16 GEMV activation exceeds shared memory capacity");
        bytes = p.k * 2;
        if (bytes + static_bytes > 48 * 1024) {
            static const auto status = cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic);
            CUDA_CHECK(status);
        }
    }
    kernel<<<p.rows / Schedule::kBlockRows, Schedule::kThreads, bytes, stream>>>(
        p.x, p.weight, output, epilogue, p.k);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule, class Output, class Epilogue>
void launch_bf16_a16_simt(const Bf16A16Operands& p, Output output, Epilogue epilogue,
                          cudaStream_t stream) {
    validate_bf16_operands<Schedule>(p);
    if (p.rows % Schedule::kBlockRows ||
        p.k % (bf16_predicated_k<Schedule>
                   ? (Schedule::kValuesPerLane < 8 ? 8 : Schedule::kValuesPerLane)
                   : Schedule::kWarpsPerRow * 32 * Schedule::kValuesPerLane) ||
        (Schedule::kTokenCapacity && p.tokens > Schedule::kTokenCapacity) ||
        (Schedule::kExactTokens && p.tokens != Schedule::kTokenCapacity))
        throw std::invalid_argument("BF16 SIMT requires complete row/K tiles and matching tokens");
    if constexpr (requires { Epilogue::kRowTokens; }) {
        if (p.tokens != Epilogue::kRowTokens || p.tokens > Schedule::kBlockTokens)
            throw std::invalid_argument("BF16 row epilogue requires the complete token interval");
    }
    common::for_each_token_slice(p.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const dim3 grid(p.rows / Schedule::kBlockRows,
                        common::div_up(count, Schedule::kBlockTokens));
        bf16_a16_simt_kernel<Schedule><<<grid, Schedule::kThreads, 0, stream>>>(
            p.x, p.weight, output, epilogue, p.k, p.tokens, offset);
        CUDA_CHECK(cudaGetLastError());
    });
}

template <class Schedule, class Output, class Epilogue>
void launch_bf16_a16_mma(const Bf16A16Operands& p, Output output, Epilogue epilogue,
                         cudaStream_t stream) {
    launch_bf16_mma_partitions<Schedule, 1>(p, output, epilogue, stream);
}

template <class Schedule, class Output, class Epilogue>
void launch_bf16_a16_sliced_k_mma(const Bf16A16Operands& p, Output output, Epilogue epilogue,
                                  cudaStream_t stream) {
    validate_bf16_operands<Schedule>(p);
    if ((!bf16_predicated_rows<Schedule> && p.rows % Schedule::kBlockRows) ||
        p.k % (bf16_predicated_k<Schedule> ? 8 : Schedule::kBlockK))
        throw std::invalid_argument("BF16 sliced-K requires compatible row/K tiles");
    constexpr auto kernel = bf16_a16_sliced_k_mma_kernel<Schedule, Output, Epilogue>;
    const int bytes       = common::prepare_shared<Schedule::kSharedBytes, kernel, true>();
    common::for_each_token_slice(p.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const dim3 grid(bf16_predicated_rows<Schedule>
                            ? common::div_up(p.rows, Schedule::kBlockRows)
                            : p.rows / Schedule::kBlockRows,
                        common::div_up(count, Schedule::kBlockTokens));
        kernel<<<grid, Schedule::kThreads, bytes, stream>>>(p.x, p.weight, output, epilogue, p.rows,
                                                            p.k, p.tokens, offset);
        CUDA_CHECK(cudaGetLastError());
    });
}

} // namespace ninfer::ops::detail
