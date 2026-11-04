#pragma once
#include "core/layout.h"
#include <cuda_runtime_api.h>
#include "core/weight.h"
#include <algorithm>
#include <span>

namespace ninfer::ops::detail {

// Scratch lasts until the stream finishes this projection. No persistent
// expansion or allocation occurs inside the CUDA implementation.
struct VoltaLinearScratch {
    void* activations = nullptr;
    void* metadata    = nullptr;
    void* weights     = nullptr;
    float* results    = nullptr;
};

inline constexpr int kVoltaTokenCapacity = 2048;

inline constexpr int volta_row_capacity(int rows) { return rows > 34816 ? 8192 : rows; }

inline constexpr std::size_t kVoltaActivationTileBytes = 528;

template <class Allocator>
VoltaLinearScratch allocate_volta_linear(Allocator& arena, int n, int k, int tokens) {
    tokens = std::min(tokens, kVoltaTokenCapacity);
    n      = volta_row_capacity(n);
    VoltaLinearScratch out;
    out.activations = arena.alloc_bytes(std::size_t(k) * tokens * 2, 256).data;
    out.metadata    = arena.alloc_bytes(std::size_t(tokens) * kVoltaActivationTileBytes, 256).data;
    if (tokens >= 128) {
        out.weights = arena.alloc_bytes(std::size_t(n) * k * 2, 256).data;
        out.results = static_cast<float*>(arena.alloc_bytes(std::size_t(n) * tokens * 4, 256).data);
    }
    return out;
}

inline std::size_t volta_linear_capacity(int n, int k, int tokens) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_volta_linear(layout, n, k, tokens);
    return layout.peak_bytes(1);
}

void volta_linear_projection(const Tensor&, const Weight&, std::span<const Tensor>,
                             VoltaLinearScratch, cudaStream_t);
void volta_linear_add(const Tensor&, const Weight&, Tensor&, VoltaLinearScratch, cudaStream_t);
void volta_linear_swiglu(const Tensor&, const Weight&, Tensor&, VoltaLinearScratch, cudaStream_t);
} // namespace ninfer::ops::detail
