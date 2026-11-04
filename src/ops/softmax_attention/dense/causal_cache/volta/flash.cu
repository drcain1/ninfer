// Adapted from Infernix and geoffwatts/ninfer-v100; see third_party/volta/NOTICE.
#include "fattn-mma-f16.cuh"
#include "ops/softmax_attention/dense/causal_cache/volta/flash.h"
#include "ops/kv_cache/kernels/layout.cuh"
#include "ops/kv_cache/kernels/hadamard.cuh"
#include "core/device.h"
#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
constexpr int kDKQ = 256;
constexpr int kDV  = 256;

// The kernel picks ncols2 from the GQA ratio, and this table mirrors that choice
// per registered geometry:
//   24q/4kv -> ratio 6: 6 % 8 != 0, 6 % 4 != 0, then DKQ <= 256 and 6 % 2 == 0
//                       -> ncols2 = 2
//   16q/2kv -> ratio 8: 8 % 8 == 0                              -> ncols2 = 8
// The 24-head geometry uses 32 query columns (64 total with GQA) to amortize
// K/V loads and FP32 PV accumulation. The 16-head geometry retains 32 total columns.
// A causal mask tensor is mandatory in both cases: mask == nullptr is only legal
// at ncols2 == 1.
template <typename Geometry>
struct VoltaFlashTiling;

template <>
struct VoltaFlashTiling<CausalD256H24Kv4> {
    static constexpr int ncols2 = 2;
    static constexpr int ncols1 = 32;
};

template <>
struct VoltaFlashTiling<CausalD256H16Kv2> {
    static constexpr int ncols2 = 8;
    static constexpr int ncols1 = 4;
};

template <typename Geometry>
struct VoltaFlashParams {
    static constexpr int kQHeads  = Geometry::QHeads;
    static constexpr int kKVHeads = Geometry::KVHeads;
    static constexpr int kGroup   = Geometry::GroupSize;
    static constexpr int kNcols2  = VoltaFlashTiling<Geometry>::ncols2;
    static constexpr int kNcols1  = VoltaFlashTiling<Geometry>::ncols1;
    static constexpr int kNcols   = kNcols1 * kNcols2;

    static_assert(kQHeads / kKVHeads == kGroup, "gqa ratio mismatch");
    static_assert(kGroup % kNcols2 == 0, "ncols2 must divide the GQA group");
    static_assert(kNcols >= 32, "Volta compiles no tile smaller than 32 columns");
};

// The key extent handed to the kernel is padded to FATTN_KQ_STRIDE. Upstream
// always satisfies this (llama.cpp pads its KV cache to 256 whenever FA is on), so
// the kernel's last key tile is free to read a whole tile past the true key count.
// Without the padding those reads would run off the end of a mask row and into the
// next one, which a multiple-of-32 key count happens to hide.
constexpr int kKeyPad = FATTN_KQ_STRIDE;

constexpr int round_up_keys(int n) { return ((n + kKeyPad - 1) / kKeyPad) * kKeyPad; }

// Reduce the device query positions once before gathering. The cached entry
// permits arbitrary query order, so its last row need not have the largest position.
__global__ void prepare_extent(CausalAttentionOperands p, const int* valid_columns, int* last) {
    __shared__ int maxima[8];
    const int valid = valid_columns ? valid_columns[0] : p.width;
    int maximum     = -1;
    for (int token = threadIdx.x; token < valid; token += blockDim.x)
        maximum = max(maximum, p.positions[token]);
    for (int step = 16; step; step /= 2)
        maximum = max(maximum, __shfl_down_sync(0xffffffffu, maximum, step));
    if (threadIdx.x % 32 == 0) maxima[threadIdx.x / 32] = maximum;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int warp = 1; warp < 8; ++warp) maximum = max(maximum, maxima[warp]);
        *last = maximum;
    }
}

// Record tables contain byte offsets shared by all planes, not page indices.
// Only valid query positions determine readable cache entries.
template <int Heads>
__global__ void gather(Int8KvReadView cache, CausalAttentionOperands p, half* keys, half* values,
                       int* exceptional, const int* last_position) {
    const int key = blockIdx.x, head = blockIdx.y, d = threadIdx.x;
    const auto dst = std::int64_t(key) * Heads * 256 + head * 256 + d;
    if (key >= p.visible_capacity || key > *last_position) {
        keys[dst] = values[dst] = __float2half(0.f);
        return;
    }
    const int table_row   = cache.table_rows ? cache.table_rows[0] : 0;
    const auto record     = cache.tables[std::int64_t(table_row) * cache.table_stride + key / 64];
    const auto code       = kv::Int8G64Layout::code_offset(record, head, d, key % 64);
    const auto scale      = kv::Int8G64Layout::scale_offset(record, head, d / 64, key % 64);
    const float key_value = float(cache.keys[code]) * __half2float(cache.key_scales[scale]);
    const float value     = float(cache.values[code]) * __half2float(cache.value_scales[scale]);
    const bool unsafe = !isfinite(key_value) || !isfinite(value) || fabsf(key_value) > 65504.f ||
                        fabsf(value) > 65504.f;
    const bool any_unsafe = __syncthreads_or(unsafe);
    if (any_unsafe && threadIdx.x == 0) atomicExch(exceptional, 1);
    keys[dst]   = __float2half(key_value);
    values[dst] = __float2half(value);
}

__global__ void prepare_query(CausalAttentionOperands p, const int* valid_columns, float* out,
                              int begin, int tokens, int* exceptional) {
    const int row = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= tokens * p.query_heads) return;
    const int valid = valid_columns ? valid_columns[0] : p.width;
    float values[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
        values[i] =
            begin + row / p.query_heads < valid
                ? __bfloat162float(
                      p.q[(std::int64_t(begin) * p.query_heads + row) * 256 + lane + 32 * i])
                : 0.f;
    kv::normalized_hadamard_d256_inplace(values, lane);
    bool unsafe = false;
#pragma unroll
    for (float value : values) unsafe |= !isfinite(value) || fabsf(value) > 65504.f;
    const bool any_unsafe = __any_sync(0xffffffffu, unsafe);
    if (any_unsafe && lane == 0) atomicExch(exceptional, 1);
#pragma unroll
    for (int i = 0; i < 8; ++i) out[std::int64_t(row) * 256 + lane + 32 * i] = values[i];
}

__global__ void prepare_mask(CausalAttentionOperands p, const int* valid_columns, half* mask,
                             int begin, int tokens, int padded, int* key_limits, int query_columns,
                             int key_tile) {
    const int row   = blockIdx.x;
    const int valid = valid_columns ? valid_columns[0] : p.width;
    // Bound each query tile by its last causally visible key. Positions need not
    // be consecutive or sorted; invalid query tails never contribute a limit.
    if (threadIdx.x == 0 && row < tokens && row % query_columns == 0) {
        int last_key = 0;
        for (int t = row; t < min(row + query_columns, tokens); ++t)
            if (begin + t < valid) last_key = max(last_key, p.positions[begin + t]);
        key_limits[row / query_columns] = min(padded, (last_key + key_tile) / key_tile * key_tile);
    }
    // Dummy rows attend to the zero-filled first key when every column is invalid;
    // their output is explicitly zeroed below, avoiding an all-minus-infinity softmax.
    const int last = row < tokens && begin + row < valid ? p.positions[begin + row] : 0;
    for (int key = threadIdx.x; key < padded; key += blockDim.x)
        mask[std::int64_t(row) * padded + key] = __float2half(key <= last ? 0.f : -INFINITY);
}

__global__ void finish(CausalAttentionOperands p, const int* valid_columns, const float* in,
                       int begin, int tokens) {
    const auto i = std::int64_t(blockIdx.x) * 256 + threadIdx.x;
    if (i >= std::int64_t(tokens) * p.query_heads * 256) return;
    const int valid = valid_columns ? valid_columns[0] : p.width;
    p.out[std::int64_t(begin) * p.query_heads * 256 + i] =
        __float2bfloat16(begin + i / (p.query_heads * 256) < valid ? in[i] : 0.f);
}

// The tensor-core path stages FP16 operands. Rare rows outside that range use
// the represented BF16 query and INT8/FP16 cache directly, without a host read.
// FP64 also keeps the Hadamard transform and score reduction finite for large Q.
template <int Heads>
__global__ void range_fallback(CausalAttentionOperands p, Int8KvReadView cache,
                               const int* exceptional, int begin, int tokens) {
    if (!*exceptional) return;
    const int token = begin + blockIdx.x, head = blockIdx.y, d = threadIdx.x;
    const int valid = cache.valid_columns ? cache.valid_columns[0] : p.width;
    if (blockIdx.x >= tokens) return;
    const auto output = (std::int64_t(token) * p.query_heads + head) * 256 + d;
    if (token >= valid) {
        p.out[output] = __float2bfloat16(0.f);
        return;
    }
    __shared__ double query[256], sums[8], alpha, beta, denominator, maximum;
    query[d] = double(__bfloat162float(p.q[output]));
    __syncthreads();
#pragma unroll
    for (int stride = 1; stride < 256; stride *= 2) {
        const double value = query[d], peer = query[d ^ stride];
        __syncthreads();
        query[d] = (d & stride) ? peer - value : value + peer;
        __syncthreads();
    }
    const double q = query[d] * 0.0625;
    if (d == 0) {
        denominator = 0.;
        maximum     = -INFINITY;
    }
    const int table_row = cache.table_rows ? cache.table_rows[0] : 0;
    const int kv_head   = head / (p.query_heads / Heads);
    double result       = 0.;
    for (int key = 0; key <= p.positions[token]; ++key) {
        const auto record = cache.tables[std::int64_t(table_row) * cache.table_stride + key / 64];
        const auto code   = kv::Int8G64Layout::code_offset(record, kv_head, d, key % 64);
        const auto scale  = kv::Int8G64Layout::scale_offset(record, kv_head, d / 64, key % 64);
        double dot = q * double(cache.keys[code]) * double(__half2float(cache.key_scales[scale]));
#pragma unroll
        for (int step = 16; step; step /= 2) dot += __shfl_down_sync(0xffffffffu, dot, step);
        if (d % 32 == 0) sums[d / 32] = dot;
        __syncthreads();
        if (d == 0) {
            double score = 0.;
#pragma unroll
            for (int warp = 0; warp < 8; ++warp) score += sums[warp];
            score *= double(p.scale);
            const double next = fmax(maximum, score);
            alpha             = exp(maximum - next);
            beta              = exp(score - next);
            denominator       = denominator * alpha + beta;
            maximum           = next;
        }
        __syncthreads();
        result = result * alpha + beta * double(cache.values[code]) *
                                      double(__half2float(cache.value_scales[scale]));
        // Every lane must consume the broadcast coefficients before the next key.
        __syncthreads();
    }
    p.out[output] = __float2bfloat16(float(result / denominator));
}

struct FlashLaunchConfig {
    int nthreads         = 0;
    int nwarps           = 0;
    int nbatch_fa        = 0;
    size_t nbytes_shared = 0;
    int blocks_per_sm    = 0;
    int nsm              = 0;
};

// One cached config per geometry: the shared-memory and occupancy figures depend
// only on the tiling, so they are resolved once per instantiation.
template <typename Geometry>
const FlashLaunchConfig& flash_launch_config() {
    using P = VoltaFlashParams<Geometry>;

    static const FlashLaunchConfig config = [] {
        FlashLaunchConfig c;

        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        c.nsm = prop.multiProcessorCount;

        const int cc = prop.major * 100 + prop.minor * 10;

        const int nbatch_K2      = ggml_cuda_fattn_mma_get_nbatch_K2(kDKQ, kDV, P::kNcols, cc);
        const int nbatch_V2      = ggml_cuda_fattn_mma_get_nbatch_V2(kDKQ, kDV, P::kNcols, cc);
        const int nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(kDKQ, kDV, P::kNcols, cc);
        const bool Q_in_reg      = ggml_cuda_fattn_mma_get_Q_in_reg(kDKQ, kDV, P::kNcols, cc);
        const int nstages = ggml_cuda_fattn_mma_get_nstages(kDKQ, kDV, P::kNcols1, P::kNcols2, cc);

        c.nthreads  = ggml_cuda_fattn_mma_get_nthreads(kDKQ, kDV, P::kNcols, cc);
        c.nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa(kDKQ, kDV, P::kNcols, cc);
        c.nwarps    = c.nthreads / WARP_SIZE;

        const int cols_per_warp = std::min(P::kNcols, get_cols_per_warp(cc));

        const size_t shared_KV_1stage =
            size_t(c.nbatch_fa) * std::max(nbatch_K2 + 4, nbatch_V2 + 4) * sizeof(half2);
        const size_t shared_KV_2stage =
            size_t(c.nbatch_fa) * (nbatch_K2 + 4 + nbatch_V2 + 4) * sizeof(half2);
        const size_t shared_Q    = size_t(P::kNcols) * (kDKQ / 2 + 4) * sizeof(half2);
        const size_t shared_mask = size_t(P::kNcols1) * (c.nbatch_fa / 2 + 4) * sizeof(half2);
        const size_t shared_combine =
            size_t(c.nwarps) * cols_per_warp * (nbatch_combine + 4) * sizeof(half2);
        const size_t shared_KV = nstages <= 1 ? shared_KV_1stage : shared_KV_2stage;

        c.nbytes_shared =
            std::max(shared_combine, Q_in_reg ? std::max(shared_Q, shared_KV + shared_mask)
                                              : shared_Q + shared_KV + shared_mask);

        auto kernel = flash_attn_ext_f16<kDKQ, kDV, P::kNcols1, P::kNcols2, false, false>;
        CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel),
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        c.nbytes_shared));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &c.blocks_per_sm, reinterpret_cast<const void*>(kernel), c.nthreads, c.nbytes_shared));
        if (c.blocks_per_sm <= 0) { c.blocks_per_sm = 1; }
        return c;
    }();
    return config;
}

// One Q-block through the vendored kernel plus its stream-K fixup.
template <typename Geometry>
void launch_flash_block(const float* q_f32, const half* k_f16, const half* v_f16, const half* mask,
                        float* dst, float2* dst_meta, const int* key_limits, int tokens, int n_kv,
                        int mask_stride, float scale, cudaStream_t stream) {
    using P                = VoltaFlashParams<Geometry>;
    constexpr int kQHeads  = P::kQHeads;
    constexpr int kKVHeads = P::kKVHeads;
    constexpr int kGroup   = P::kGroup;
    constexpr int kNcols1  = P::kNcols1;
    constexpr int kNcols2  = P::kNcols2;

    const FlashLaunchConfig& c = flash_launch_config<Geometry>();

    const int ntiles_x     = (tokens + kNcols1 - 1) / kNcols1;
    const int ntiles_z_gqa = (kGroup + kNcols2 - 1) / kNcols2;
    const int ntiles_dst   = ntiles_x * ntiles_z_gqa * kKVHeads;
    const int ntiles_KV    = (n_kv + c.nbatch_fa - 1) / c.nbatch_fa;

    const int max_blocks = c.blocks_per_sm * c.nsm;

    const int raw     = std::min(max_blocks, ntiles_KV * ntiles_dst);
    const int rounded = (raw / ntiles_dst) * ntiles_dst;
    const int loss    = rounded > 0 ? 100 * (raw - rounded) / raw : 100;
    const int nblocks = loss <= 5 ? rounded : raw;

    const dim3 block_dim(WARP_SIZE, c.nwarps, 1);
    const dim3 blocks_num(nblocks, 1, 1);

    const std::int32_t nb01 = kDKQ * kQHeads * sizeof(float);
    const std::int32_t nb02 = kDKQ * sizeof(float);
    const std::int32_t nb11 = kDKQ * kKVHeads * sizeof(half);
    const std::int32_t nb12 = kDKQ * sizeof(half);
    const std::int32_t nb31 = mask_stride * sizeof(half);

    const uint3 ne01_fd = init_fastdiv_values(tokens);

    flash_attn_ext_f16<kDKQ, kDV, kNcols1, kNcols2, false, false>
        <<<blocks_num, block_dim, c.nbytes_shared, stream>>>(
            reinterpret_cast<const char*>(q_f32), reinterpret_cast<const char*>(k_f16),
            reinterpret_cast<const char*>(v_f16), reinterpret_cast<const char*>(mask), nullptr,
            key_limits, dst, dst_meta, scale, 0.0f, 1.0f, 1.0f, /*n_head_log2=*/16u, 0.0f, kDKQ,
            ne01_fd, kQHeads, 1, nb01, nb02, 0, kDKQ, n_kv, kKVHeads, 1, nb11, nb12, 0, nb11, nb12,
            0, tokens, 1, 1, nb31, 0, 0);

    if (nblocks % ntiles_dst == 0 && nblocks > ntiles_dst) {
        const uint3 fd0 = init_fastdiv_values(ntiles_x * ntiles_z_gqa * kKVHeads);
        const uint3 fd1 = init_fastdiv_values(ntiles_x * ntiles_z_gqa);
        const uint3 fd2 = init_fastdiv_values(ntiles_x);

        flash_attn_stream_k_fixup_uniform<kDV, kNcols1, kNcols2>
            <<<dim3((unsigned)ntiles_dst, kNcols1, kNcols2), dim3(kDV, 1, 1), 0, stream>>>(
                dst, dst_meta, tokens, kQHeads, kKVHeads, nblocks, kGroup, nblocks / ntiles_dst,
                fd0, fd1, fd2);
    } else if (ntiles_dst % nblocks != 0) {
        const int total_work = ntiles_KV * ntiles_dst;

        const uint3 fd_k_j_z_ne12 =
            init_fastdiv_values(ntiles_KV * ntiles_x * ntiles_z_gqa * kKVHeads);
        const uint3 fd_k_j_z = init_fastdiv_values(ntiles_KV * ntiles_x * ntiles_z_gqa);
        const uint3 fd_k_j   = init_fastdiv_values(ntiles_KV * ntiles_x);
        const uint3 fd_k     = init_fastdiv_values(ntiles_KV);

        flash_attn_stream_k_fixup_general<kDV, kNcols1, kNcols2>
            <<<dim3((unsigned)nblocks, kNcols1, kNcols2), dim3(kDV, 1, 1), 0, stream>>>(
                dst, dst_meta, tokens, kQHeads, kGroup, total_work, fd_k_j_z_ne12, fd_k_j_z, fd_k_j,
                fd_k);
    }
}

// ---------------------------------------------------------------------------

template <class G>
void execute(const CausalAttentionOperands& p, Int8KvReadView cache, VoltaFlashScratch s,
             cudaStream_t stream) {
    const int padded  = round_up_keys(p.visible_capacity);
    auto* exceptional = static_cast<int*>(s.exceptional.data);
    CUDA_CHECK(cudaMemsetAsync(exceptional, 0, sizeof(int), stream));
    prepare_extent<<<1, 256, 0, stream>>>(p, cache.valid_columns,
                                          static_cast<int*>(s.key_limits.data));
    gather<G::KVHeads><<<dim3(padded, G::KVHeads), 256, 0, stream>>>(
        cache, p, static_cast<half*>(s.key.data), static_cast<half*>(s.value.data), exceptional,
        static_cast<const int*>(s.key_limits.data));
    for (int begin = 0; begin < p.width; begin += kVoltaFlashQBlockTokens) {
        const int tokens = std::min(kVoltaFlashQBlockTokens, p.width - begin);
        const int rows   = (tokens + 63) / 64 * 64;
        const auto count = std::int64_t(tokens) * p.query_heads * 256;
        prepare_query<<<(tokens * p.query_heads + 7) / 8, 256, 0, stream>>>(
            p, cache.valid_columns, static_cast<float*>(s.query.data), begin, tokens, exceptional);
        prepare_mask<<<rows, 256, 0, stream>>>(
            p, cache.valid_columns, static_cast<half*>(s.mask.data), begin, tokens, padded,
            static_cast<int*>(s.key_limits.data), VoltaFlashParams<G>::kNcols1,
            flash_launch_config<G>().nbatch_fa);
        CUDA_CHECK(cudaMemsetAsync(s.output.data, 0, count * sizeof(float), stream));
        launch_flash_block<G>(static_cast<float*>(s.query.data), static_cast<half*>(s.key.data),
                              static_cast<half*>(s.value.data), static_cast<half*>(s.mask.data),
                              static_cast<float*>(s.output.data), static_cast<float2*>(s.meta.data),
                              static_cast<int*>(s.key_limits.data), tokens, padded, padded, p.scale,
                              stream);
        finish<<<(count + 255) / 256, 256, 0, stream>>>(
            p, cache.valid_columns, static_cast<float*>(s.output.data), begin, tokens);
        range_fallback<G::KVHeads>
            <<<dim3(tokens, p.query_heads), 256, 0, stream>>>(p, cache, exceptional, begin, tokens);
    }
    CUDA_CHECK(cudaGetLastError());
}
} // namespace

void volta_flash_attention(const CausalAttentionOperands& p, Int8KvReadView cache,
                           MemoryArena& workspace, int multiprocessor_count, cudaStream_t stream) {
    auto scope   = workspace.scope();
    auto scratch = allocate_volta_flash(workspace, p.query_heads, p.width, p.visible_capacity,
                                        multiprocessor_count);
    if (p.query_heads == 24)
        execute<CausalD256H24Kv4>(p, cache, scratch, stream);
    else
        execute<CausalD256H16Kv2>(p, cache, scratch, stream);
}
} // namespace ninfer::ops::detail
