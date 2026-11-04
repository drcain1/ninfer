# llama.cpp Volta MMA flash-attention (vendored)

Vendored from [llama.cpp](https://github.com/ggml-org/llama.cpp), MIT licensed
(see `LICENSE`), pinned at commit **`62bf73d25`**. Same provenance pattern as
`src/ops/common/volta_mma.cuh`, which was transcribed from this project's
`ggml-cuda/mma.cuh`.

Used by the sm_70 prefill attention route. Imported through the NInfer V100 fork
recorded in `../volta/NOTICE`, then adapted to Infernix workspace and numerical contracts.

## Contents

| file | origin | change |
|---|---|---|
| `mma.cuh` | `ggml/src/ggml-cuda/mma.cuh` | added Volta row/row FP32 accumulation |
| `cp-async.cuh` | `ggml/src/ggml-cuda/cp-async.cuh` | none (byte-for-byte) |
| `fattn-mma-f16.cuh` | `ggml/src/ggml-cuda/fattn-mma-f16.cuh` | truncated, Volta config, FP32 PV accumulation |
| `fattn-stream-k.cuh` | `ggml/src/ggml-cuda/fattn-common.cuh` lines 721-971 | extracted verbatim |
| `common.cuh` | shim, not upstream | see below |

## Deviations from upstream

**`common.cuh` is ours.** Upstream's is 1,669 lines and pulls in `ggml.h`,
`ggml-impl.h`, `ggml-cuda.h` and `ggml-common.h`. The device side of the kernel
needs about a dozen symbols from it, none involving a ggml type. Providing them
under the upstream file name avoids a dependency on ggml model and runtime types. Definitions inside are copied verbatim from
upstream except where a comment marks otherwise.

**`fattn-mma-f16.cuh` is truncated at line 1893**, dropping the host-side
`ggml_cuda_flash_attn_ext_mma_f16_case` launcher and its `DECL_` instantiation
macros, which take `ggml_tensor` / `ggml_backend_cuda_context`. Infernix drives
the kernel from its own launcher. Its `#include "fattn-common.cuh"` becomes
`#include "fattn-stream-k.cuh"`; nothing in the device kernel referenced that
header. `fattn-common.cuh` itself is not vendored, so `vecdotq.cuh` and
`convert.cuh` (quantized-KV only) drop out of the closure entirely.

**One added config case.** `ggml_cuda_fattn_mma_get_config_volta` has no
256/256 entry upstream and falls through to the Ampere table's
`nbatch_combine = 128`. On Volta `get_cols_per_warp()` is 32 (16 on Turing+),
making the combine buffer 66 KB and capping the kernel at 1 block/SM on a
96 KB/SM V100. Every genuine Volta entry upstream uses 64; giving 256/256 the
same reduces the combine buffer. Occupancy also depends on the accumulator
register footprint. It sits under upstream's own
`// TODO tune specifically for Volta`.

**FP32 PV accumulation.** Infernix uses Volta `mma.sync.m8n8k4.row.row.f32`
for the weighted-value product, with FP32 rescaling between key tiles. The donor's
FP16 accumulators failed the BF16 attention oracle on long contexts. Tile extents,
row mapping and reduction staging were updated together. The public BF16, INT8 and
FP8 attention tests qualify the resulting route against independently decoded KV.

## Updating

Re-vendor from a single upstream commit and re-run the correctness and throughput harness before
trusting the result. Keep the pin in this file in sync.
