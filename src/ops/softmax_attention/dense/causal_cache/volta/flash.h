#pragma once
#include "core/layout.h"
#include "ops/softmax_attention/kernels/causal/operands.h"
#include <algorithm>

namespace ninfer::ops::detail {
inline constexpr int kVoltaFlashQBlockTokens = 512;

inline std::size_t volta_flash_meta_elements(int heads, int tokens, int multiprocessor_count) {
    const int columns       = heads == 24 ? 64 : 32;
    const int query_columns = heads == 24 ? 32 : 4;
    const int kv_heads      = heads == 24 ? 4 : 2;
    const int groups        = heads == 24 ? 3 : 1;
    const int tiles         = ((tokens + query_columns - 1) / query_columns) * groups * kv_heads;
    // Every launch reserves at least its padded query tile in shared memory.
    // V100 has 96 KiB per SM. Ignoring register limits gives a safe occupancy
    // upper bound without device queries during host-only workspace planning.
    const int query_shared_bytes = columns * (256 / 2 + 4) * 4;
    const auto blocks            = std::max(std::size_t(tiles), std::size_t(multiprocessor_count) *
                                                                    ((96 * 1024) / query_shared_bytes));
    return blocks * columns * (2 + 256 / 2);
}

struct VoltaFlashScratch {
    Tensor key, value, mask, query, output, meta, exceptional, key_limits;
};

template <class Allocator>
VoltaFlashScratch allocate_volta_flash(Allocator& arena, int heads, int width, int keys,
                                       int multiprocessor_count) {
    const int kv_heads = heads == 24 ? 4 : 2;
    const int padded   = (keys + 255) / 256 * 256;
    const int tokens   = std::min(width, kVoltaFlashQBlockTokens);
    return {arena.alloc(DType::FP16, {256, kv_heads, padded}),
            arena.alloc(DType::FP16, {256, kv_heads, padded}),
            arena.alloc(DType::FP16, {padded, (tokens + 63) / 64 * 64}),
            arena.alloc(DType::FP32, {256, heads, tokens}),
            arena.alloc(DType::FP32, {256, heads, tokens}),
            arena.alloc(DType::FP32,
                        {2, int(volta_flash_meta_elements(heads, tokens, multiprocessor_count))}),
            arena.alloc(DType::I32, {1}),
            arena.alloc(DType::I32, {(tokens + 3) / 4})};
}

inline std::size_t volta_flash_workspace_bytes(int heads, int width, int keys,
                                               int multiprocessor_count) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_volta_flash(layout, heads, width, keys, multiprocessor_count);
    return layout.peak_bytes(1);
}

void volta_flash_attention(const CausalAttentionOperands&, Int8KvReadView, MemoryArena&,
                           int multiprocessor_count, cudaStream_t);
} // namespace ninfer::ops::detail
