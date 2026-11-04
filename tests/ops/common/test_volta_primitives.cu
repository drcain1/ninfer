#include "ops/common/cuda/mma.cuh"
#include "ops/linear/kernels/volta/tiled.cuh"
#include "core/pdl.cuh"
#include "ops/common/numeric/convert.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/quant/nvfp4.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace ninfer::ops::common;
using ninfer::ops::Cache;

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

__global__ void loads(unsigned* out) {
    __shared__ unsigned short matrix[4][8][8];
    int lane = threadIdx.x;
    for (int i = lane; i < 256; i += 32) reinterpret_cast<unsigned short*>(matrix)[i] = i + 1;
    __syncthreads();
    unsigned address = smem_addr(&matrix[lane / 8][lane % 8][0]);
    unsigned r[4];
    ldmatrix_x4(r[0], r[1], r[2], r[3], address);
    for (int m = 0; m < 4; ++m) out[lane * 8 + m] = r[m];
    ldmatrix_x4_t(r[0], r[1], r[2], r[3], address);
    for (int m = 0; m < 4; ++m) out[lane * 8 + 4 + m] = r[m];
}

// Input fragments are supplied by the host. The reference below computes full
// dense matrix products from decoded storage, independently of warp exchanges.
template <int Kind>
__global__ void product(const unsigned* input, float* output) {
    int lane          = threadIdx.x;
    const unsigned* p = input + lane * 8;
    float c[4]        = {0.25f, -0.5f, 0.75f, -1.f};
    for (int iteration = 0; iteration < 2; ++iteration) {
        if constexpr (Kind == 0)
            mma_bf16(c[0], c[1], c[2], c[3], p[0], p[1], p[2], p[3], p[4], p[5]);
        if constexpr (Kind == 1)
            mma_f16(c[0], c[1], c[2], c[3], p[0], p[1], p[2], p[3], p[4], p[5]);
        if constexpr (Kind == 2)
            mma_tf32_bits(c[0], c[1], c[2], c[3], p[0], p[1], p[2], p[3], p[4], p[5]);
        if constexpr (Kind == 3)
            mma_fp8_e4m3(c[0], c[1], c[2], c[3], p[0], p[1], p[2], p[3], p[4], p[5]);
        if constexpr (Kind == 4)
            mma_fp8_e4m3_scaled(c[0], c[1], c[2], c[3], p[0], p[1], p[2], p[3], p[4], p[5], p[6],
                                p[7]);
        if constexpr (Kind == 5)
            mma_nvfp4_e4m3(c[0], c[1], c[2], c[3], p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
    }
    for (int i = 0; i < 4; ++i) output[lane * 4 + i] = c[i];
}

__global__ void integer_product(const unsigned* input, int* output) {
    int lane      = threadIdx.x;
    const auto* p = input + lane * 8;
    int c0 = 1000000000, c1 = 2000000000, c2 = -1234567890, c3 = 2147483647;
    mma_s8(c0, c1, c2, c3, p[0], p[1], p[2], p[3], p[4], p[5]);
    output[lane * 4]     = c0;
    output[lane * 4 + 1] = c1;
    output[lane * 4 + 2] = c2;
    output[lane * 4 + 3] = c3;
}

float fp8(unsigned v) {
    int e = (v >> 3) & 15, m = v & 7;
    float x = e ? std::ldexp(1.f + m / 8.f, e - 7) : std::ldexp(float(m), -9);
    return v & 128 ? -x : x;
}

float fp4(unsigned v) {
    constexpr float t[] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    return v & 8 ? -t[v & 7] : t[v & 7];
}

struct Fixture {
    int bits, k;
    std::vector<unsigned> a, b, packed;
    std::vector<double> da, db;

    Fixture(int kind, bool wide = false)
        : bits(kind == 2   ? 32
               : kind >= 3 ? kind == 5 ? 4 : 8
                           : 16),
          k(256 / bits), a(16 * k), b(k * 8), packed(32 * 8), da(a.size()), db(b.size()) {
        auto encode = [&](int index, bool left, unsigned& code, double& decoded) {
            float x = float((index * 13 + (left ? 7 : 3)) % 29 - 14) / 8;
            if (wide) x *= left ? 1.e20f : 1.e-20f;
            if (kind == 0) {
                auto v  = __float2bfloat16(x);
                code    = __bfloat16_as_ushort(v);
                decoded = __bfloat162float(v);
            } else if (kind == 1) {
                auto v  = __float2half(x);
                code    = __half_as_ushort(v);
                decoded = __half2float(v);
            } else if (kind == 2) {
                union {
                    float f;
                    unsigned u;
                } v{x};
                code    = v.u;
                decoded = x;
            } else if (kind == 5) {
                code    = (index * 7 + 3) % 16;
                decoded = fp4(code);
            } else {
                code    = ((index * 5) % 32 + 40) | ((index % 3 == 0) ? 128 : 0);
                decoded = fp8(code);
            }
        };
        for (int i = 0; i < int(a.size()); ++i) encode(i, true, a[i], da[i]);
        for (int i = 0; i < int(b.size()); ++i) encode(i, false, b[i], db[i]);
        int per = 32 / bits;
        for (int lane = 0; lane < 32; ++lane) {
            int row = lane / 4, offset = (lane % 4) * per;
            for (int r = 0; r < 4; ++r)
                for (int j = 0; j < per; ++j)
                    packed[lane * 8 + r] |=
                        a[(row + (r % 2) * 8) * k + offset + (r / 2) * (k / 2) + j] << (j * bits);
            for (int r = 0; r < 2; ++r)
                for (int j = 0; j < per; ++j)
                    packed[lane * 8 + 4 + r] |= b[(offset + r * (k / 2) + j) * 8 + row]
                                                << (j * bits);
            if (kind == 4) {
                packed[lane * 8 + 6] = 125 + (row + (lane % 2) * 8) % 5;
                packed[lane * 8 + 7] = 126 + row % 3;
            }
            if (kind == 5)
                for (int j = 0; j < 4; ++j) {
                    packed[lane * 8 + 6] |= (48 + (row + (lane % 2) * 8 + j) % 12) << (j * 8);
                    packed[lane * 8 + 7] |= (48 + (row + 2 * j) % 12) << (j * 8);
                }
        }
        for (int r = 0; r < 16; ++r)
            for (int q = 0; q < k; ++q) {
                if (kind == 4) da[r * k + q] *= std::ldexp(1., 125 + r % 5 - 127);
                if (kind == 5) da[r * k + q] *= fp8(48 + (r + q / 16) % 12);
            }
        for (int q = 0; q < k; ++q)
            for (int c = 0; c < 8; ++c) {
                if (kind == 4) db[q * 8 + c] *= std::ldexp(1., 126 + c % 3 - 127);
                if (kind == 5) db[q * 8 + c] *= fp8(48 + (c + 2 * (q / 16)) % 12);
            }
    }
};

template <int Kind>
void verify(unsigned* device_input, float* device_output, bool wide = false) {
    Fixture f(Kind, wide);
    check(cudaMemcpy(device_input, f.packed.data(), f.packed.size() * 4, cudaMemcpyHostToDevice));
    product<Kind><<<1, 32>>>(device_input, device_output);
    check(cudaGetLastError());
    std::array<float, 128> out;
    check(cudaMemcpy(out.data(), device_output, 512, cudaMemcpyDeviceToHost));
    constexpr double initial[] = {.25, -.5, .75, -1};
    for (int lane = 0; lane < 32; ++lane)
        for (int i = 0; i < 4; ++i) {
            int row = lane / 4 + (i / 2) * 8, col = (lane % 4) * 2 + i % 2;
            double expected = initial[i], magnitude = 1;
            for (int q = 0; q < f.k; ++q) {
                double v = 2 * f.da[row * f.k + q] * f.db[q * 8 + col];
                expected += v;
                magnitude += std::abs(v);
            }
            if (!std::isfinite(out[lane * 4 + i]) ||
                std::abs(out[lane * 4 + i] - expected) > 2.e-6 * magnitude)
                throw std::runtime_error("MMA independent oracle mismatch, format " +
                                         std::to_string(Kind));
        }
}

__global__ void represented_values(float* values) {
    const unsigned code = threadIdx.x;
    values[code]        = volta::decode<volta::Format::E4m3>(code);
    if (code < 16) values[256 + code] = volta::decode<volta::Format::E2m1>(code);
}

__global__ void packed_fp8_values(float2* values) {
    const unsigned pair = blockIdx.x * blockDim.x + threadIdx.x;
    values[pair] = __half22float2(ninfer::ops::detail::volta_fp8_pair(pair & 255, pair >> 8));
}

void verify_packed_fp8() {
    constexpr int count = 256 * 256;
    float2* output      = nullptr;
    check(cudaMalloc(&output, count * sizeof(float2)));
    packed_fp8_values<<<count / 256, 256>>>(output);
    check(cudaGetLastError());
    std::vector<float2> actual(count);
    check(cudaMemcpy(actual.data(), output, count * sizeof(float2), cudaMemcpyDeviceToHost));
    check(cudaFree(output));
    for (unsigned pair = 0; pair < count; ++pair) {
        for (int lane = 0; lane < 2; ++lane) {
            const unsigned code = (pair >> (lane * 8)) & 255;
            const float value   = lane ? actual[pair].y : actual[pair].x;
            if ((code & 127) == 127) {
                if (!std::isnan(value)) throw std::runtime_error("packed FP8 NaN mismatch");
            } else if (value != fp8(code) || std::signbit(value) != bool(code & 128)) {
                throw std::runtime_error("packed FP8 represented value mismatch");
            }
        }
    }
}

void verify_represented_values() {
    float* output = nullptr;
    check(cudaMalloc(&output, 272 * sizeof(float)));
    represented_values<<<1, 256>>>(output);
    check(cudaGetLastError());
    std::array<float, 272> actual{};
    check(cudaMemcpy(actual.data(), output, sizeof(actual), cudaMemcpyDeviceToHost));
    check(cudaFree(output));
    for (unsigned code = 0; code < 256; ++code) {
        if ((code & 127) == 127) {
            if (!std::isnan(actual[code])) throw std::runtime_error("FP8 NaN decode mismatch");
        } else if (actual[code] != fp8(code) || std::signbit(actual[code]) != bool(code & 128)) {
            throw std::runtime_error("FP8 represented value mismatch");
        }
    }
    constexpr float fp4[] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
    for (unsigned code = 0; code < 16; ++code) {
        const float expected = (code & 8) ? -fp4[code & 7] : fp4[code & 7];
        if (actual[256 + code] != expected || std::signbit(actual[256 + code]) != bool(code & 8))
            throw std::runtime_error("FP4 represented value mismatch");
    }
}

__global__ void reductions(int* output) {
    const int lane             = threadIdx.x;
    constexpr unsigned masks[] = {0xffffffffu, 0x0000ffffu, 0x55555555u, 0x80000001u};
    for (int i = 0; i < 4; ++i) {
        const unsigned mask = masks[i];
        if (mask & (1u << lane)) {
            output[i * 96 + lane * 3]     = masked_warp_max(mask, lane - 31);
            output[i * 96 + lane * 3 + 1] = masked_warp_min(mask, lane - 31);
            output[i * 96 + lane * 3 + 2] = masked_warp_or(mask, 1u << lane);
        }
    }
}

void verify_reductions() {
    int* output = nullptr;
    check(cudaMalloc(&output, 4 * 96 * sizeof(int)));
    reductions<<<1, 32>>>(output);
    check(cudaGetLastError());
    std::array<int, 4 * 96> actual{};
    check(cudaMemcpy(actual.data(), output, sizeof(actual), cudaMemcpyDeviceToHost));
    check(cudaFree(output));
    constexpr unsigned masks[] = {0xffffffffu, 0x0000ffffu, 0x55555555u, 0x80000001u};
    for (int i = 0; i < 4; ++i) {
        int lo = 31, hi = 0;
        for (int lane = 0; lane < 32; ++lane)
            if (masks[i] & (1u << lane)) {
                lo = std::min(lo, lane);
                hi = std::max(hi, lane);
            }
        for (int lane = 0; lane < 32; ++lane)
            if (masks[i] & (1u << lane)) {
                const int* got = actual.data() + i * 96 + lane * 3;
                if (got[0] != hi - 31 || got[1] != lo - 31 || unsigned(got[2]) != masks[i])
                    throw std::runtime_error("masked warp reduction mismatch");
            }
    }
}

__global__ void copies_and_quantization(const unsigned* input, unsigned* output) {
    __shared__ unsigned char staging[64 * 16];
    const int lane     = threadIdx.x;
    const auto* source = reinterpret_cast<const unsigned char*>(input) + lane * 16;
    auto* destination  = staging + lane * 16;
    const int valid    = lane % 17;
    if (lane % 3 == 0) cp_async_zfill<16>(destination, source, valid);
    if (lane % 3 == 1)
        cp_async_cache_hint<Cache::cg>(destination, source, valid, l2_streaming_policy());
    if (lane % 3 == 2) cp_async_zfill<16, Cache::cg>(destination, source, valid);
    cp_commit();
    cp_wait<0>();
    __syncthreads();
    // Cross-warp reads exercise publication after the caller's CTA barrier.
    const int other                        = (lane + 32) % 64;
    reinterpret_cast<uint4*>(output)[lane] = load_vec<uint4>(staging + other * 16);
}

__global__ void quantization(unsigned* output) {
    // Include every FP4 rounding midpoint, positive and negative.
    constexpr float midpoint[] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.f, 7.f};
    float2 values[8];
    for (int i = 0; i < 8; ++i) values[i] = make_float2(midpoint[i], -midpoint[i]);
    pack_nvfp4_e2m1x16(values, output[0], output[1]);
    output[2] = pack_bf16x2(1.00390625f, -1.01171875f);
}

void verify_copies_and_quantization(unsigned* input, float* output) {
    std::array<unsigned char, 1024> source{}, actual{};
    for (int i = 0; i < 1024; ++i) source[i] = (i * 11 + 7) % 251;
    check(cudaMemcpy(input, source.data(), 1024, cudaMemcpyHostToDevice));
    copies_and_quantization<<<1, 64>>>(input, reinterpret_cast<unsigned*>(output));
    check(cudaGetLastError());
    check(cudaMemcpy(actual.data(), output, 1024, cudaMemcpyDeviceToHost));
    for (int lane = 0; lane < 64; ++lane)
        for (int byte = 0; byte < 16; ++byte) {
            const int other         = (lane + 32) % 64;
            const unsigned expected = byte < other % 17 ? source[other * 16 + byte] : 0;
            if (actual[lane * 16 + byte] != expected)
                throw std::runtime_error("synchronous pipeline copy mismatch");
        }
    quantization<<<1, 1>>>(input);
    check(cudaGetLastError());
    std::array<unsigned, 3> codes{};
    check(cudaMemcpy(codes.data(), input, sizeof(codes), cudaMemcpyDeviceToHost));
    constexpr unsigned rounded[] = {0, 2, 2, 4, 4, 6, 6, 7};
    for (int i = 0; i < 8; ++i) {
        const unsigned code     = (codes[i / 4] >> (8 * (i % 4))) & 255;
        const unsigned expected = rounded[i] | ((rounded[i] | 8) << 4);
        if (code != expected) throw std::runtime_error("FP4 ties-to-even or saturation mismatch");
    }
    if (codes[2] != 0xbf823f80u) throw std::runtime_error("BF16 ties-to-even mismatch");
}

__global__ void dependency_producer(unsigned* values) {
    if (threadIdx.x == 0) ninfer::pdl::trigger_dependents();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    values[i] += 1;
}

__global__ void dependency_consumer(const unsigned* values, unsigned* output) {
    ninfer::pdl::wait_for_dependencies();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    output[i]   = values[i] * 2;
}

void verify_dependencies(unsigned* values, float* output) {
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::array<unsigned, 256> initial{}, actual{};
    for (int i = 0; i < 256; ++i) initial[i] = i * 3;
    check(cudaMemcpy(values, initial.data(), sizeof(initial), cudaMemcpyHostToDevice));
    const auto launch = [&] {
        dependency_producer<<<4, 64, 0, stream>>>(values);
        check(cudaGetLastError());
        check(ninfer::pdl::launch_dependent({dim3(4), dim3(64), 0, stream}, dependency_consumer,
                                            values, reinterpret_cast<unsigned*>(output)));
    };
    const auto verify = [&](unsigned executions) {
        check(cudaStreamSynchronize(stream));
        check(cudaMemcpy(actual.data(), output, sizeof(actual), cudaMemcpyDeviceToHost));
        for (int i = 0; i < 256; ++i)
            if (actual[i] != (initial[i] + executions) * 2)
                throw std::runtime_error("dependent launch ordering mismatch");
    };
    launch();
    verify(1);
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    launch();
    check(cudaStreamEndCapture(stream, &graph));
    check(cudaGraphInstantiate(&executable, graph, 0));
    check(cudaGraphLaunch(executable, stream));
    verify(2);
    check(cudaGraphLaunch(executable, stream));
    verify(3);
    check(cudaGraphExecDestroy(executable));
    check(cudaGraphDestroy(graph));
    check(cudaStreamDestroy(stream));
}

// Rows select either a large or tiny input coordinate. Each output is checked
// exactly, so a large reference norm cannot hide lost small BF16 values.
template <bool Nvfp4, bool Tiled, int Tiles = 2>
void verify_volta_activation_range() {
    using namespace ninfer::ops::detail;
    constexpr int n = 128, k = 64, tokens = 33;
    std::vector<__nv_bfloat16> input(k * tokens), output(n * tokens);
    std::vector<unsigned char> codes(n * k / (Nvfp4 ? 2 : 1), 0);
    std::vector<unsigned char> nv_scales(n * k / 16, 56); // E4M3 1
    std::vector<__nv_bfloat16> fp_scales(n, __float2bfloat16(1.f));
    for (int row = 0; row < n; ++row) {
        const int column = row % 2 ? 3 : 0;
        if constexpr (Nvfp4)
            codes[row * k / 2 + column / 2] = 2u << ((column % 2) * 4);
        else
            codes[row * k + column] = 56; // E4M3 1
    }
    __nv_bfloat16 *x = nullptr, *y = nullptr;
    unsigned char *weights = nullptr, *scales = nullptr;
    check(cudaMalloc(&x, input.size() * 2));
    check(cudaMalloc(&y, output.size() * 2));
    check(cudaMalloc(&weights, codes.size()));
    const std::size_t scale_bytes = Nvfp4 ? nv_scales.size() : fp_scales.size() * 2;
    check(cudaMalloc(&scales, scale_bytes));
    check(cudaMemcpy(weights, codes.data(), codes.size(), cudaMemcpyHostToDevice));
    check(cudaMemcpy(scales,
                     Nvfp4 ? static_cast<const void*>(nv_scales.data())
                           : static_cast<const void*>(fp_scales.data()),
                     scale_bytes, cudaMemcpyHostToDevice));
    const auto launch = [&] {
        const auto invoke = [&](auto operands) {
            const LinearBf16Output destination{y, n};
            if constexpr (Tiled)
                volta_tiled_kernel<false><<<dim3(n / 32, (tokens + 31) / 32), 128>>>(
                    operands, destination, LinearIdentityEpilogue{});
            else
                volta_qpn_kernel<Tiles, false>
                    <<<dim3(n / 32, (tokens + Tiles * 8 - 1) / (Tiles * 8)), 128>>>(
                        operands, destination, LinearIdentityEpilogue{});
        };
        if constexpr (Nvfp4)
            invoke(Nvfp4A16Operands{x, weights, scales, n, k, tokens, 1.f});
        else
            invoke(
                Fp8A16Operands{x, weights, reinterpret_cast<__nv_bfloat16*>(scales), n, k, tokens});
        check(cudaGetLastError());
    };
    // Input mutation exercises exact admission, sparse correction and the
    // dense-residual FP32 fallback using the actual production kernels.
    for (int mode = 0; mode < 8; ++mode) {
        std::fill(input.begin(), input.end(), __float2bfloat16(0.f));
        for (int token = 0; token < tokens; ++token) {
            if (mode == 2) {
                for (int col = 0; col < k; ++col)
                    input[token * k + col] = __float2bfloat16(token % 2 ? 1.e-30f : 1.e15f);
            } else if (mode < 2) {
                input[token * k]     = __float2bfloat16(mode == 0 ? 1.f : 1.e15f);
                input[token * k + 3] = __float2bfloat16(1.e-30f);
            } else if (mode == 3 || mode == 4) {
                // The low BF16 mantissa bit is exactly representable at the
                // FP16 subnormal boundary, and needs correction one bit below.
                input[token * k] = __float2bfloat16(1.f);
                input[token * k + 3] =
                    __float2bfloat16(std::ldexp(1.0078125f, mode == 3 ? -32 : -33));
            } else if (mode == 5) {
                input[token * k]     = __ushort_as_bfloat16(0x7f7f); // largest finite BF16
                input[token * k + 3] = __ushort_as_bfloat16(1);      // smallest BF16 subnormal
            } else if (mode == 6) {
                input[token * k]     = __ushort_as_bfloat16(1);
                input[token * k + 3] = __ushort_as_bfloat16(3);
            }
        }
        check(cudaMemcpy(x, input.data(), input.size() * 2, cudaMemcpyHostToDevice));
        launch();
        check(cudaMemcpy(output.data(), y, output.size() * 2, cudaMemcpyDeviceToHost));
        for (int token = 0; token < tokens; ++token)
            for (int row = 0; row < n; ++row)
                if (__bfloat16_as_ushort(output[token * n + row]) !=
                    __bfloat16_as_ushort(input[token * k + (row % 2 ? 3 : 0)]))
                    throw std::runtime_error("Volta activation range/correction mismatch");
    }
    check(cudaFree(x));
    check(cudaFree(y));
    check(cudaFree(weights));
    check(cudaFree(scales));
}

int main() {
    try {
        unsigned* in = nullptr;
        float* out   = nullptr;
        check(cudaMalloc(&in, 1024));
        check(cudaMalloc(&out, 1024));
        loads<<<1, 32>>>(reinterpret_cast<unsigned*>(out));
        check(cudaGetLastError());
        std::array<unsigned, 256> loaded;
        check(cudaMemcpy(loaded.data(), out, 1024, cudaMemcpyDeviceToHost));
        for (int lane = 0; lane < 32; ++lane)
            for (int m = 0; m < 4; ++m) {
                unsigned r = lane / 4, c = (lane % 4) * 2;
                unsigned normal = (m * 64 + r * 8 + c + 1) | ((m * 64 + r * 8 + c + 2) << 16);
                unsigned transpose =
                    (m * 64 + c * 8 + r + 1) | ((m * 64 + (c + 1) * 8 + r + 1) << 16);
                if (loaded[lane * 8 + m] != normal || loaded[lane * 8 + 4 + m] != transpose)
                    throw std::runtime_error("matrix load mismatch");
            }
        verify_volta_activation_range<false, false>();
        verify_volta_activation_range<false, true>();
        verify_volta_activation_range<true, false>();
        verify_volta_activation_range<true, false, 1>();
        verify_volta_activation_range<true, true>();
        verify<0>(in, out);
        verify<0>(in, out, true);
        verify<1>(in, out);
        verify<2>(in, out);
        verify<3>(in, out);
        verify<4>(in, out);
        verify<5>(in, out);
        std::vector<unsigned> packed(256);
        std::array<int, 128> got;
        for (int lane = 0; lane < 32; ++lane)
            for (int r = 0; r < 6; ++r)
                for (int j = 0; j < 4; ++j)
                    packed[lane * 8 + r] |= unsigned((lane * 7 + r * 11 + j * 13) % 256) << (j * 8);
        check(cudaMemcpy(in, packed.data(), 1024, cudaMemcpyHostToDevice));
        integer_product<<<1, 32>>>(in, reinterpret_cast<int*>(out));
        check(cudaGetLastError());
        check(cudaMemcpy(got.data(), out, 512, cudaMemcpyDeviceToHost));
        std::array<std::array<int, 32>, 16> a{};
        std::array<std::array<int, 8>, 32> b{};
        for (int lane = 0; lane < 32; ++lane)
            for (int r = 0; r < 6; ++r)
                for (int j = 0; j < 4; ++j) {
                    int value = static_cast<signed char>((packed[lane * 8 + r] >> (j * 8)) & 255);
                    if (r < 4)
                        a[lane / 4 + (r % 2) * 8][(lane % 4) * 4 + (r / 2) * 16 + j] = value;
                    else
                        b[(lane % 4) * 4 + (r - 4) * 16 + j][lane / 4] = value;
                }
        constexpr int initial[] = {1000000000, 2000000000, -1234567890, 2147483647};
        for (int lane = 0; lane < 32; ++lane)
            for (int i = 0; i < 4; ++i) {
                unsigned expected = unsigned(initial[i]);
                for (int q = 0; q < 32; ++q)
                    expected +=
                        unsigned(a[lane / 4 + (i / 2) * 8][q] * b[q][(lane % 4) * 2 + i % 2]);
                if (unsigned(got[lane * 4 + i]) != expected)
                    throw std::runtime_error("integer MMA mismatch");
            }
        verify_copies_and_quantization(in, out);
        verify_reductions();
        verify_represented_values();
        verify_packed_fp8();
        verify_dependencies(in, out);
        check(cudaFree(in));
        check(cudaFree(out));
        std::puts(
            "Volta matrix, pipeline, quantization and independent dense-product oracles passed");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
