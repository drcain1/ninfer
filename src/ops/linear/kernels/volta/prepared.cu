#include "ops/linear/kernels/volta/prepared.h"
#include "ops/linear/kernels/volta/prepared_qpn.cuh"
#include "ops/linear/kernels/volta/prepared_cutlass.cuh"
#include "ops/linear/swiglu/paired_output.cuh"

namespace ninfer::ops::detail {
namespace {
static_assert(sizeof(VoltaActivationTile) == kVoltaActivationTileBytes);

template <int Count>
struct SegmentedOutput {
    __nv_bfloat16* data[Count]{};
    int rows[Count]{}, offset = 0, row_offset = 0;

    __device__ __forceinline__ void store(int row, int token, float value) const {
        row += row_offset;
#pragma unroll
        for (int i = 0; i < Count; ++i) {
            if (row < rows[i]) {
                data[i][std::int64_t(token + offset) * rows[i] + row] = __float2bfloat16_rn(value);
                return;
            }
            row -= rows[i];
        }
    }
};

template <bool Paired, class Output, class Epilogue>
void launch(const Tensor& x, const Weight& w, Output output, Epilogue epilogue,
            VoltaLinearScratch scratch, cudaStream_t stream) {
    VoltaPreparedActivation prepared{static_cast<half*>(scratch.activations),
                                     static_cast<VoltaActivationTile*>(scratch.metadata)};
    auto invoke = [&](auto p) {
        if (p.tokens >= 128) {
            launch_volta_prepared_gemm<Paired>(p, output, epilogue, prepared,
                                               static_cast<half*>(scratch.weights), scratch.results,
                                               stream);
        } else {
            auto small = [&](auto tiles) {
                constexpr int Tiles = decltype(tiles)::value;
                volta_prepare_activation<1>
                    <<<p.tokens, 128, 0, stream>>>(p.x, p.k, p.tokens, prepared);
                const auto schedule = [&](auto warps) {
                    constexpr int Warps = decltype(warps)::value;
                    volta_prepared_qpn<Tiles, Warps, 1, Paired>
                        <<<dim3((p.rows + 31) / 32, (p.tokens + Tiles * 8 - 1) / (Tiles * 8)),
                           Warps * 32, 0, stream>>>(p, output, epilogue, prepared);
                };
                if constexpr (std::is_same_v<decltype(p), VoltaPackedFp8>) {
                    schedule(std::integral_constant<int, 4>{});
                } else if constexpr (std::is_same_v<decltype(p), VoltaPackedNvfp4> && Tiles == 1) {
                    // Small output grids need more K warps to occupy the V100.
                    if (p.rows <= 8192 && p.k >= 16384)
                        schedule(std::integral_constant<int, 16>{});
                    else
                        schedule(std::integral_constant<int, 8>{});
                } else {
                    schedule(std::integral_constant<int, 8>{});
                }
            };
            if (p.tokens <= 8)
                small(std::integral_constant<int, 1>{});
            else
                small(std::integral_constant<int, 2>{});
        }
    };
    if (w.layout == QuantLayout::VoltaQpn) {
        if (w.qtype == QType::NVFP4)
            invoke(VoltaPackedNvfp4{nvfp4_a16_operands(x, w)});
        else
            invoke(VoltaPackedFp8{fp8_a16_operands(x, w)});
    } else if (w.qtype == QType::NVFP4)
        invoke(nvfp4_a16_operands(x, w));
    else
        invoke(fp8_a16_operands(x, w));
    CUDA_CHECK(cudaGetLastError());
}
} // namespace

template <int Count>
void project_outputs(const Tensor& x, const Weight& w, std::span<const Tensor> outputs,
                     VoltaLinearScratch scratch, cudaStream_t stream) {
    SegmentedOutput<Count> output;
    for (int i = 0; i < Count; ++i) {
        output.data[i] = static_cast<__nv_bfloat16*>(outputs[i].data);
        output.rows[i] = outputs[i].ne[0];
    }
    const int row_capacity = x.ne[1] >= 128 ? volta_row_capacity(w.n) : w.n;
    for (int first = 0; first < w.n; first += row_capacity) {
        Weight weight     = w;
        weight.n          = std::min(w.n - first, row_capacity);
        output.row_offset = first;
        // Only the registered full-vocabulary FP8 matrix needs row tiling.
        if (first) {
            weight.qdata  = static_cast<const unsigned char*>(w.qdata) + std::int64_t(first) * w.k;
            weight.scales = static_cast<const __nv_bfloat16*>(w.scales) + first;
        }
        for (int begin = 0; begin < x.ne[1]; begin += kVoltaTokenCapacity) {
            const Tensor tile = x.slice(1, begin, std::min(x.ne[1] - begin, kVoltaTokenCapacity));
            output.offset     = begin;
            launch<false>(tile, weight, output, LinearIdentityEpilogue{}, scratch, stream);
        }
    }
}

void volta_linear_projection(const Tensor& x, const Weight& w, std::span<const Tensor> outputs,
                             VoltaLinearScratch scratch, cudaStream_t stream) {
    switch (outputs.size()) {
    case 1:
        return project_outputs<1>(x, w, outputs, scratch, stream);
    case 2:
        return project_outputs<2>(x, w, outputs, scratch, stream);
    case 3:
        return project_outputs<3>(x, w, outputs, scratch, stream);
    case 4:
        return project_outputs<4>(x, w, outputs, scratch, stream);
    default:
        throw std::invalid_argument("Volta projection requires one to four output segments");
    }
}

void volta_linear_add(const Tensor& x, const Weight& w, Tensor& out, VoltaLinearScratch scratch,
                      cudaStream_t stream) {
    for (int begin = 0; begin < x.ne[1]; begin += kVoltaTokenCapacity) {
        const int count   = std::min(x.ne[1] - begin, kVoltaTokenCapacity);
        const Tensor tile = x.slice(1, begin, count);
        auto* data        = static_cast<__nv_bfloat16*>(out.data) + std::int64_t(begin) * w.n;
        launch<false>(tile, w, LinearBf16Output{data, w.n}, LinearResidualAddEpilogue{{data, w.n}},
                      scratch, stream);
    }
}

void volta_linear_swiglu(const Tensor& x, const Weight& w, Tensor& out, VoltaLinearScratch scratch,
                         cudaStream_t stream) {
    for (int begin = 0; begin < x.ne[1]; begin += kVoltaTokenCapacity) {
        const int count   = std::min(x.ne[1] - begin, kVoltaTokenCapacity);
        const Tensor tile = x.slice(1, begin, count);
        auto* data        = static_cast<__nv_bfloat16*>(out.data) + std::int64_t(begin) * (w.n / 2);
        launch<true>(tile, w, LinearBf16Output{data, w.n / 2}, SwiGluPairEpilogue{}, scratch,
                     stream);
    }
}
} // namespace ninfer::ops::detail
