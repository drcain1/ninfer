#include "artifact/encoded_slice.h"
#include "artifact/fixture.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"

#include <bit>
#include <cstring>
#include <cuda_runtime.h>

namespace ninfer::test {
namespace {
using namespace artifact;
using namespace artifact_fixture;

void selection(QType format, QuantLayout layout, const char* name, const char* encoded_layout,
               std::optional<CudaExecutionView> execution) {
    Fixture f;
    const bool nv     = format == QType::NVFP4;
    const bool bank   = layout == QuantLayout::BatchedBlockScaleK16M128x4;
    const Shape shape = bank ? Shape{2, 384, 256} : Shape{384, 256};
    const auto g      = weight_geometry(format, layout, shape);
    f.payload.resize(g.bytes);
    for (std::size_t i = 0; i < f.payload.size(); ++i)
        f.payload[i] = std::byte((i * 37 + i / 73) % 251);
    for (std::uint64_t i = 0; i < g.divisor_count; ++i)
        put_word(f.payload, g.divisor_offset + i * 4, std::bit_cast<std::uint32_t>(2.5F + i), 4);
    f.root["components"] = {{"text", {{"config", Json::object()}}}};
    f.root["objects"]    = Json::array({{{"id", "source"},
                                         {"kind", "tensor"},
                                         {"shape", shape},
                                         {"format", name},
                                         {"layout", encoded_layout},
                                         {"offset", 0},
                                         {"bytes", g.bytes}}});
    f.root["bindings"]   = {{"source", {{"object", "source"}}}};
    f.root["uses"]       = Json::array();
    f.root["files"]      = Json::array({{{"path", nullptr}, {"payload_bytes", 19001}},
                                        {{"path", "second.bin"}, {"payload_bytes", g.bytes - 19001}}});
    f.write();
    Reader reader(f.entry);
    const auto object = reader.find("source");
    const auto origin = bank ? 384U : 0U;
    const std::array slices{MatrixSlice{object, origin + 256, 128, 128, 128, 0, 0},
                            MatrixSlice{object, origin, 128, 128, 128, 128, 0}};
    const Shape target_shape{256, 128};
    const auto tensor = slice_tensor(reader, target_shape, slices);
    std::vector<TensorRequest> requests{{tensor, Device::cpu()},
                                        {tensor, Device::cpu()},
                                        {whole_tensor(reader, object), Device::cpu()},
                                        {tensor, Device::cpu(), HostAccess::CpuOnly, 7}};
    if (execution) requests.push_back({tensor, execution->device()});
    auto plan     = plan_materialization(reader, requests);
    const auto id = plan.requests[0];
    require(plan.requests[1] == id && plan.requests[3] != id &&
                (!execution || plan.requests[4] == id),
            "selection storage deduplication");
    auto backing       = materialize(reader, std::move(plan), execution);
    const auto& parent = backing.parent(id, Device::cpu());
    const auto& d      = parent.geometry;
    const auto equal   = [&](std::uint64_t destination, std::uint64_t source, std::uint64_t bytes) {
        require(std::memcmp(parent.data + destination, f.payload.data() + source, bytes) == 0,
                  "selected encoded words differ from source coordinates");
    };
    for (std::uint64_t row = 0; row < 256; ++row) {
        const auto src = origin + (row < 128 ? row + 256 : row - 128);
        if (layout == QuantLayout::Contiguous) {
            const auto word = g.bytes / g.elements;
            equal(row * 128 * word, (src * 256 + 128) * word, 128 * word);
            continue;
        }
        equal(row * d.code_bytes_per_row, src * g.code_bytes_per_row + g.code_bytes_per_row / 2,
              d.code_bytes_per_row);
        if (g.high_bytes)
            equal(d.high_offset + row * d.high_bytes_per_row,
                  g.high_offset + src * g.high_bytes_per_row + g.high_bytes_per_row / 2,
                  d.high_bytes_per_row);
        if (nv) {
            // Independent encoding equation: 128x4 scale tiles, row-major tile ordering.
            for (std::uint64_t group = 0; group < 8; ++group) {
                const auto offset = [](std::uint64_t r, std::uint64_t k, std::uint64_t groups) {
                    return ((r / 128) * (groups / 4) + k / 4) * 512 + (r % 32) * 16 +
                           ((r % 128) / 32) * 4 + k % 4;
                };
                equal(d.scale_offset + offset(row, group, 8),
                      g.scale_offset + offset(src, group + 8, 16), 1);
            }
        } else if (layout == QuantLayout::RowScale) {
            equal(d.scale_offset + row * 2, g.scale_offset + src * 2, 2);
        } else {
            const auto block = layout == QuantLayout::BlockScaleM128K128 ? 128 : 1;
            equal(d.scale_offset + row / block * d.scale_row_bytes,
                  g.scale_offset + src / block * g.scale_row_bytes + g.scale_row_bytes / 2,
                  d.scale_row_bytes);
        }
    }
    if (nv) {
        equal(d.divisor_offset, g.divisor_offset + (bank ? 4 : 0), 4);
        require(parent.weight_divisors[0] == (bank ? 3.5F : 2.5F), "selected divisor metadata");
    }
    if (execution) {
        std::vector<std::byte> actual(d.bytes);
        CUDA_CHECK(cudaMemcpy(actual.data(), backing.parent(id, execution->device()).data,
                              actual.size(), cudaMemcpyDeviceToHost));
        require(std::memcmp(actual.data(), parent.data, d.bytes) == 0,
                "selected GPU/CPU encoded bytes differ");
    }
    auto invalid               = slices;
    invalid[1].destination_row = 0;
    rejects([&] { (void)slice_tensor(reader, target_shape, invalid); },
            "overlapping destination accepted");
}

void expert_shards(bool nvfp4, std::optional<CudaExecutionView> execution) {
    // I=640 exercises the 320-row gate/up seam and K=320 down slices. Distinct per-expert
    // divisors and scale words detect cross-expert or cross-block aliasing.
    for (const bool down : {false, true}) {
        Fixture f;
        const Shape shape = down ? Shape{2, 256, 640} : Shape{2, 1280, 256};
        const auto g = weight_geometry(nvfp4 ? QType::NVFP4 : QType::FP8_E4M3FN_BLOCK128X128_BF16,
                                       nvfp4 ? QuantLayout::BatchedBlockScaleK16M128x4
                                             : QuantLayout::BlockScaleM128K128,
                                       shape);
        f.payload.resize(g.bytes);
        for (std::size_t i = 0; i < f.payload.size(); ++i)
            f.payload[i] = std::byte((i * 37 + i / 73) % 251);
        for (std::uint64_t e = 0; e < g.divisor_count; ++e)
            put_word(f.payload, g.divisor_offset + e * 4, std::bit_cast<std::uint32_t>(2.5F + e),
                     4);
        f.root["components"] = {{"text", {{"config", Json::object()}}}};
        f.root["objects"]    = Json::array(
            {{{"id", "source"},
                 {"kind", "tensor"},
                 {"shape", shape},
                 {"format", nvfp4 ? "nvfp4" : "fp8_e4m3fn_block128x128_bf16"},
                 {"layout", nvfp4 ? "batched_block_scale_k16_m128x4_v1" : "block_scale_m128_k128_v1"},
                 {"offset", 0},
                 {"bytes", g.bytes}}});
        f.root["bindings"] = {{"source", {{"object", "source"}}}};
        f.root["uses"]     = Json::array();
        f.root["files"]    = Json::array({{{"path", nullptr}, {"payload_bytes", g.bytes}}});
        f.write();
        Reader reader(f.entry);
        const auto object = reader.find("source");
        if (nvfp4 && !down) {
            const std::array mixed_divisors{MatrixSlice{object, 0, 0, 320, 256, 0, 0},
                                            MatrixSlice{object, 1920, 0, 320, 256, 320, 0}};
            rejects([&] { (void)slice_tensor(reader, Shape{640, 256}, mixed_divisors); },
                    "NVFP4 assembly accepted different expert divisors in one matrix");
        }
        for (std::uint64_t rank = 0; rank < 2; ++rank) {
            std::vector<MatrixSlice> slices;
            for (std::uint64_t e = 0; e < 2; ++e) {
                if (down)
                    slices.push_back({object, e * 256, rank * 320, 256, 320, e * 256, 0});
                else
                    for (std::uint64_t half = 0; half < 2; ++half)
                        slices.push_back({object, e * 1280 + half * 640 + rank * 320, 0, 320, 256,
                                          e * 640 + half * 320, 0});
            }
            const auto tensor =
                slice_tensor(reader, down ? Shape{2, 256, 320} : Shape{2, 640, 256}, slices);
            require(nvfp4 || tensor.geometry.format == QType::FP8_E4M3FN_BLOCK64X64_BF16,
                    "FP8 slice did not refine its scale blocks");
            std::vector<TensorRequest> requests{{tensor, Device::cpu()}};
            if (execution) requests.push_back({tensor, execution->device()});
            auto plan          = plan_materialization(reader, requests);
            const auto id      = plan.requests[0];
            auto backing       = materialize(reader, std::move(plan), execution);
            const auto& parent = backing.parent(id, Device::cpu());
            const auto& d      = parent.geometry;
            const auto equal   = [&](std::uint64_t destination, std::uint64_t source,
                                   std::size_t bytes) {
                require(std::memcmp(parent.data + destination, f.payload.data() + source, bytes) ==
                              0,
                          "expert shard changed represented codes/scales/divisors");
            };
            const auto scale = [](const WeightGeometry& g, std::uint64_t row,
                                  std::uint64_t column) {
                if (g.format == QType::NVFP4) {
                    const auto group = column / 16;
                    return g.scale_offset + (row / 128 * (g.shape.back() / 64) + group / 4) * 512 +
                           row % 32 * 16 + row % 128 / 32 * 4 + group % 4;
                }
                const auto block = g.format == QType::FP8_E4M3FN_BLOCK64X64_BF16 ? 64U : 128U;
                return g.scale_offset +
                       (row / block * (g.shape.back() / block) + column / block) * 2;
            };
            const auto word_divisor = nvfp4 ? 2U : 1U;
            for (std::uint64_t row = 0; row < d.elements / d.shape.back(); ++row) {
                const auto source_row =
                    down ? row : row / 640 * 1280 + row % 640 / 320 * 640 + row % 320 + rank * 320;
                const auto column = down ? rank * 320 : 0;
                equal(row * d.code_bytes_per_row,
                      source_row * g.code_bytes_per_row + column / word_divisor,
                      d.shape.back() / word_divisor);
                for (std::uint64_t k = 0; k < d.shape.back(); k += nvfp4 ? 16 : 64)
                    equal(scale(d, row, k), scale(g, source_row, column + k), nvfp4 ? 1 : 2);
            }
            if (nvfp4) {
                for (std::uint64_t e = 0; e < 2; ++e) {
                    equal(d.divisor_offset + e * 4, g.divisor_offset + e * 4, 4);
                    require(parent.weight_divisors[e] == 2.5F + e, "local expert divisor differs");
                }
            }
            if (execution) {
                std::vector<std::byte> actual(d.bytes);
                CUDA_CHECK(cudaMemcpy(actual.data(), backing.parent(id, execution->device()).data,
                                      actual.size(), cudaMemcpyDeviceToHost));
                require(std::memcmp(actual.data(), parent.data, d.bytes) == 0,
                        "expert shard GPU upload differs");
            }
        }
    }
}

void assembly_and_padding(std::optional<CudaExecutionView> execution) {
    Fixture f;
    f.payload.assign(192, std::byte{});
    for (std::uint64_t i = 0; i < 96; ++i) put_word(f.payload, i * 2, i, 2);
    f.root["components"] = {{"text", {{"config", Json::object()}}}};
    f.root["objects"]    = Json::array({{{"id", "conv"},
                                         {"kind", "tensor"},
                                         {"shape", {4, 24}},
                                         {"format", "bf16"},
                                         {"layout", "contiguous_le_v1"},
                                         {"offset", 0},
                                         {"bytes", 192}}});
    f.root["bindings"]   = {{"conv", {{"object", "conv"}}}};
    f.root["uses"]       = Json::array();
    f.root["files"]      = Json::array({{{"path", nullptr}, {"payload_bytes", 77}},
                                        {{"path", "second.bin"}, {"payload_bytes", 115}}});
    f.write();
    Reader reader(f.entry);
    const auto object = reader.find("conv");
    const std::array slices{MatrixSlice{object, 0, 2, 4, 2, 0, 0},
                            MatrixSlice{object, 0, 6, 4, 2, 0, 2},
                            MatrixSlice{object, 0, 16, 4, 8, 0, 4}};
    const auto selected = slice_tensor(reader, Shape{4, 12}, slices);
    const auto device   = execution ? execution->device() : Device::cpu();
    const std::array requests{TensorRequest{selected, device}};
    auto plan     = plan_materialization(reader, requests);
    const auto id = plan.requests[0];
    auto backing  = materialize(reader, std::move(plan), execution);
    std::array<std::uint16_t, 48> words{};
    const auto* source = backing.parent(id, device).data;
    if (execution)
        CUDA_CHECK(cudaMemcpy(words.data(), source, sizeof(words), cudaMemcpyDeviceToHost));
    else
        std::memcpy(words.data(), source, sizeof(words));
    const std::array channels{2, 3, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
    for (unsigned tap = 0; tap < 4; ++tap)
        for (unsigned channel = 0; channel < 12; ++channel)
            require(words[tap * 12 + channel] == tap * 24 + channels[channel],
                    "tap-major assembly changed channel order");

    Fixture padded;
    std::fill(padded.payload.begin() + 256, padded.payload.begin() + 784, std::byte{0x5a});
    padded.write();
    Reader qr(padded.entry);
    const MatrixSlice part{qr.find("q5"), 0, 64, 2, 64};
    const auto target = slice_tensor(qr, Shape{2, 64}, std::span(&part, 1));
    const std::array request{TensorRequest{target, Device::cpu()}};
    auto qp            = plan_materialization(qr, request);
    const auto qid     = qp.requests[0];
    auto q             = materialize(qr, std::move(qp));
    const auto& parent = q.parent(qid, Device::cpu());
    const auto& g      = parent.geometry;
    for (std::uint64_t row = 0; row < 2; ++row) {
        for (const auto [base, stride, width] :
             {std::array<std::uint64_t, 3>{0, g.code_bytes_per_row, 32},
              std::array<std::uint64_t, 3>{g.high_offset, g.high_bytes_per_row, 8},
              std::array<std::uint64_t, 3>{g.scale_offset, g.scale_row_bytes, 2}}) {
            for (std::uint64_t i = 0; i < stride; ++i)
                require(parent.data[base + row * stride + i] ==
                            (i < width ? std::byte{0x5a} : std::byte{}),
                        "new quantized padding retained source words");
        }
    }
}

void volta_layout(std::optional<CudaExecutionView> execution) {
    for (const bool nv : {false, true}) {
        Fixture f;
        const Shape shape{128, 256};
        const auto format          = nv ? QType::NVFP4 : QType::FP8_E4M3FN_ROW_BF16;
        const auto g               = weight_geometry(format, QuantLayout::VoltaQpn, shape);
        const std::uint64_t codes  = 128 * 256 / (nv ? 2 : 1);
        const std::uint64_t scales = nv ? 128 * 16 : 128 * 2;
        require(g.code_bytes == codes && g.scale_offset == codes && g.scale_bytes == scales &&
                    g.bytes == codes + scales + (nv ? 4 : 0),
                "Volta packed planes changed payload extent");
        f.payload.resize(g.bytes);
        for (std::size_t i = 0; i < f.payload.size(); ++i)
            f.payload[i] = std::byte((i * 31 + i / 29) % 256);
        if (nv) put_word(f.payload, g.divisor_offset, std::bit_cast<std::uint32_t>(2.5F), 4);
        f.root["components"] = {{"text", {{"config", Json::object()}}}};
        f.root["objects"]    = Json::array({{{"id", "source"},
                                             {"kind", "tensor"},
                                             {"shape", shape},
                                             {"format", nv ? "nvfp4" : "fp8_e4m3fn_row_bf16"},
                                             {"layout", "volta_qpn_v1"},
                                             {"offset", 0},
                                             {"bytes", g.bytes}}});
        f.root["bindings"]   = {{"source", {{"object", "source"}}}};
        f.root["uses"]       = Json::array();
        f.root["files"] =
            Json::array({{{"path", nullptr}, {"payload_bytes", 1025}},
                         {{"path", "second.bin"}, {"payload_bytes", g.bytes - 1025}}});
        f.write();
        Reader reader(f.entry);
        const auto object = reader.find("source");
        std::vector<TensorRequest> requests{{whole_tensor(reader, object), Device::cpu()}};
        if (execution) requests.push_back({whole_tensor(reader, object), execution->device()});
        auto plan          = plan_materialization(reader, requests);
        const auto storage = plan.requests[0];
        auto backing       = materialize(reader, std::move(plan), execution);
        for (const auto& request : requests) {
            const auto& parent = backing.parent(storage, request.device);
            WeightView view{shape,
                            {{&parent, 0, g.elements / 2}, {&parent, g.elements / 2, g.elements}}};
            const auto weight = native_weight(view, 3.0F);
            require(weight.layout == QuantLayout::VoltaQpn && weight.qdata == parent.data &&
                        weight.scales == parent.data + codes && weight.n == 128 && weight.k == 256,
                    "native Volta weight lost its physical layout");
            require(!nv ||
                        (weight.weight_scale_divisor == 2.5F && weight.input_scale_divisor == 3.0F),
                    "Volta NVFP4 divisor changed");
            rejects<std::invalid_argument>(
                [&] { (void)native_weight(WeightView{{64, 256}, {{&parent, 0, g.elements / 2}}}); },
                "partial Volta native matrix accepted");
            rejects<std::invalid_argument>([&] { (void)weight_row_planes(view.parts[0]); },
                                           "packed fragment exposed as contiguous row");
        }
        const auto& host = backing.parent(storage, Device::cpu());
        require(std::memcmp(host.data, f.payload.data(), g.bytes) == 0,
                "Volta materialization modified encoded words");
        if (execution) {
            std::vector<std::byte> copied(g.bytes);
            CUDA_CHECK(cudaMemcpy(copied.data(), backing.parent(storage, execution->device()).data,
                                  g.bytes, cudaMemcpyDeviceToHost));
            require(copied == f.payload, "Volta GPU materialization changed encoded words");
        }
        for (std::uint64_t row = 0; row < 128; ++row)
            for (std::uint64_t group = 0; group < (nv ? 16U : 1U); ++group) {
                const auto lane   = (row & 3) + ((row >> 3) & 3) * 4 + ((row >> 2) & 1) * 16;
                const auto offset = nv ? ((row / 32 * 16 + group) * 32 + lane) : row * 2;
                require(weight_scale_offset(g, row, group) == codes + offset,
                        "Volta scale coordinate differs from physical fragment");
            }
        const std::array slice{MatrixSlice{object, 0, 0, 64, 256}};
        rejects([&] { (void)slice_tensor(reader, Shape{64, 256}, slice); },
                "packed layout silently accepted an unsupported assembly");
    }
    rejects<std::invalid_argument>(
        [] { (void)weight_geometry(QType::NVFP4, QuantLayout::VoltaQpn, Shape{127, 256}); },
        "unaligned Volta matrix accepted");
}

} // namespace

void encoded_slice_checks(std::optional<CudaExecutionView> execution) {
    volta_layout(execution);
    assembly_and_padding(execution);
    expert_shards(true, execution);
    expert_shards(false, execution);
    selection(QType::BF16, QuantLayout::Contiguous, "bf16", "contiguous_le_v1", execution);
    selection(QType::Q4_G64_FP16, QuantLayout::RowSplit, "q4_g64_fp16", "row_split_k128_v1",
              execution);
    selection(QType::Q5_G64_FP16, QuantLayout::RowSplit, "q5_g64_fp16", "row_split_k128_v1",
              execution);
    selection(QType::Q6_G64_FP16, QuantLayout::RowSplit, "q6_g64_fp16", "row_split_k128_v1",
              execution);
    selection(QType::Q8_G32_FP16, QuantLayout::RowSplit, "q8_g32_fp16", "row_split_k128_v1",
              execution);
    selection(QType::FP8_E4M3FN_ROW_BF16, QuantLayout::RowScale, "fp8_e4m3fn_row_bf16",
              "row_scale_v1", execution);
    selection(QType::FP8_E4M3FN_BLOCK128X128_BF16, QuantLayout::BlockScaleM128K128,
              "fp8_e4m3fn_block128x128_bf16", "block_scale_m128_k128_v1", execution);
    selection(QType::NVFP4, QuantLayout::BlockScaleK16M128x4, "nvfp4", "block_scale_k16_m128x4_v1",
              execution);
    selection(QType::NVFP4, QuantLayout::BatchedBlockScaleK16M128x4, "nvfp4",
              "batched_block_scale_k16_m128x4_v1", execution);
}
} // namespace ninfer::test
