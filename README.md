# NInfer Volta — full upstream V100 port

Native C++/CUDA inference for NVIDIA Tesla V100, built on the newer [NInfer](https://github.com/Neroued/ninfer) engine architecture. This fork keeps Volta hardware useful for local LLM serving while bringing it forward from the older V100 engine.

## Why this fork exists

The working V100 fork had fallen hundreds of commits behind upstream. Upstream had replaced substantial parts of the engine and targeted RTX 5090 and DGX Spark; merging those changes alone did not produce a working `sm_70` engine. We kept a conservative reliability build for daily use, then ported the newer architecture and optimized its actual V100 execution paths.

Both tracks live in **[drcain1/ninfer-volta](https://github.com/drcain1/ninfer-volta)**:

| Branch | Purpose | Validation baseline |
| --- | --- | --- |
| [`port/upstream-v100-20261103`](https://github.com/drcain1/ninfer-volta/tree/port/upstream-v100-20261103) — this branch | Full upstream engine with native Volta support and performance work | Inference commit `f4e186aa`, based on upstream `a754fcf3` |
| [`backport/upstream-reliability-20261103`](https://github.com/drcain1/ninfer-volta/tree/backport/upstream-reliability-20261103) | Older working V100 engine with selected upstream reliability fixes | Inference commit `bc3f1be2` |

These are separate engine generations with different configuration flags. The revisions above identify the tested inference code; later documentation commits do not change those binaries. As of the November 5, 2026 qualification, the full port is packaged and validated, while the reliability build remains deployed on the reference host.

## What the port does

- Runs the newer upstream engine, serving APIs, context scheduler, and prefix-cache restoration on Volta (`sm_70`).
- Adds native FP16 Volta tensor-core paths, FP32 accumulation where required by the kernels, and software handling of packed NVFP4/FP8 weights. V100 does not gain hardware FP4, FP8, or BF16 tensor cores.
- Provides lossless Volta weight-layout packing, optimized packed linear and GDN projections, and an optimized MTP proposal head.
- Accelerates INT8-KV prefill by computing all 256 value dimensions in one pass and scheduling the full query batch together. The large-value fallback remains in place.
- Reduces kernel spills and CUDA module memory, and preserves long-conversation prefix reuse under the tested host-cache budget.

The public products remain `.ninfer` model artifacts, a CLI, an OpenAI/Anthropic-compatible server, and perplexity tooling. There is no Python model-inference runtime.

## Target and qualified configuration

The qualified machine is a **Tesla V100 PCIe 32 GB**, Linux x86_64, with approximately **31 GiB host RAM**. Builds used **CUDA 12.8.1 and GCC 13**. The V100 target accepts CUDA 12.8/12.9; CUDA 13 removes `sm_70` compilation support.

End-to-end validation used **Swift 1.5 Qwen3.8 27B OrcaRouter NVFP4**, with a losslessly repacked artifact, **MTP with four draft tokens**, INT8 KV, CUDA graphs, 2,048-token prefill chunks, two execution lanes, and a 200,000-token context limit. `qwen3.8-27b` is the serving alias. Although the source artifact includes DFlash2 weights, the performance results below use MTP.

The broader upstream model implementations remain in the tree. Their presence is not a claim of V100 qualification. V100 16 GB, other Volta cards, arbitrary model families, and 232K+ contexts have not been qualified by this campaign.

## Build and run

Use CMake 3.28+, Ninja, a C++20 compiler, and CUDA 12.8 or 12.9. Install the development dependencies described in the [build-system guide](docs/maintainer/build-system.md), including libcurl, liburing, hwloc, JPEG, and FFmpeg. Select the V100 preset explicitly; upstream GPU recipes use different toolchains.

```bash
git clone --branch port/upstream-v100-20261103 https://github.com/drcain1/ninfer-volta.git
cd ninfer-volta
cmake --preset v100-release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DNINFER_ENABLE_DISTRIBUTED=OFF
cmake --build --preset v100-release --parallel 4
```

Adjust the CUDA path for your installation. The products are under `build/v100-release/apps/`; retain the accompanying `lib/` directory when packaging. `v100-core` builds only Core/Artifact validation and does **not** build the inference engine. Use `v100-dev` to enable tests; see the [test guide](tests/README.md) for selecting suites.

The newer engine needs its v3 artifact schema and bindings. Do not assume that an older `.ninfer` file is compatible just because its filename or version looks similar. See [weight conversion](docs/weight-conversion.md); the [v2 upgrade tool](tools/upgrade_ninfer_v2_to_v3.py) handles its explicitly recognized inputs. For a compatible v3 artifact, pack the Volta weight layout from the repository root:

```bash
python3 -m tools.artifact.pack_volta model-v3.ninfer model-volta-v3.ninfer
```

This changes the layout of supported packed weights, not their represented values; it does not quantize a new model. The qualification artifact was checked for exact represented-weight preservation.

The tested compute/cache profile, with an example loopback listener:

```bash
./build/v100-release/apps/ninfer-serve /path/to/model-volta-v3.ninfer \
  --host 127.0.0.1 --port 8080 --model-id qwen3.8-27b \
  --max-context 200000 --context-capacity auto --kv-dtype int8 \
  --prefill-chunk 2048 --max-concurrency 2 \
  --host-cache-capacity 9221505024B --prefix-reuse \
  --cuda-graph --spec mtp --draft-tokens 4 --proposal-head optimized \
  --preserve-thinking --vision
```

The host cache is **8.59 GiB**; it is not a cap on total process or system memory. Upstream's `io_uring` I/O path needs kernel support and permission to create its rings. The tested host uses the stock AlmaLinux/RHEL-derived `6.12.0-211.65.1.el10_2.x86_64` kernel with `kernel.io_uring_disabled=0`; no custom kernel was needed. Container syscall policy and locked-memory limits must also permit the runtime's operations.

See [CLI](docs/cli.md), [serving](docs/serving.md), and [startup configuration](docs/startup-configuration.md) for the newer engine's options. The reliability branch's `--host-kv-mib`, state-slot flags, and `--lm-head-draft` are not this branch's configuration recipe.

## What we tested

Qualification at `f4e186aa` on November 5, 2026 includes:

| Check | Result / coverage |
| --- | --- |
| Independent-reference INT8 attention | 220 cases passed, no skips; includes FP64-reference 150K history, graph updates, route boundaries, large values, masked tails, and unordered positions |
| CUDA Compute Sanitizer | memcheck, synccheck, and racecheck: three focused production cases each, no reported errors or hazards |
| MTP state correctness | Independent Main-conditioning oracle: 21 KV checks in eager mode and 21 with CUDA graphs |
| Serving regression | 16 API checks, upstream HTTP contract checks, and eight concurrent arithmetic checks passed |
| Packaged image | Binary identity, health endpoint, and a real MTP arithmetic request passed |
| Long-context replay | 49 requests per build, 21 main continuations, final prompt 180,210 tokens; labels and token counts matched |

Numerical correctness is established by independent references, rather than by whether generated text looks plausible. Earlier port validation also covered 950 linear, 180 GDN, and 140 mixed-attention cases. Those earlier suites were not all rerun for the final attention-only change.

**Known failures:** the earlier broad GPU sweep found seven pre-existing failures, also reproduced on the preceding port baseline: five QSA sparse-GQA formats, block-scaled MoE, and expert-parallel MoE (`cudaFuncSetAttribute` invalid-value failures). These are outside the qualified Qwen dense/GDN/MTP path and were not rerun for the final attention-only change. This is not a claim that the entire upstream test suite passes on V100.

## Performance and host memory

The [recorded qualification report](docs/validation/2026-11-05/REPORT.txt) compares the final port with the reliability build. Both used MTP4, INT8 KV, CUDA graphs, the same model weights, and the context/chunk/concurrency settings above. The reliability profile used 4 GiB host KV plus 32 host-state slots; the newer engine used its 8.59 GiB host cache.

| Fresh long-context replay | Full port | Reliability |
| --- | ---: | ---: |
| Cold 150K time to first token | 348.645 s | 338.918 s |
| Warm 150K–180K median time to first token | 6.858 s | 7.223 s |
| Warm maximum time to first token | 7.280 s | 7.779 s |
| Peak process RSS | 9.821 GiB | 10.142 GiB |
| Minimum host memory available | 4.048 GiB | 3.732 GiB |
| Process swap | 0 KiB | 0 KiB |

The port was 2.87% slower on the cold 150K prompt and 5.05% faster on median warm continuations. Every main continuation reused its prefix, with strictly advancing prefixes and no cache-loss multi-minute warm stalls in this replay. **Cold 150K prefill still takes nearly six minutes.** The 200K setting is a configured limit; successful replay through 180K does not establish full 200K concurrency or 232K+ support.

This was one sequential replay per build, with matched initial-temperature limits and dynamic GPU clocks. Active temperatures and clocks differed, so the small differences should not be treated as fixed-clock kernel comparisons. The port stayed within this host's measured memory budget. System-wide swapout increased by 775 pages during its run (zero during reliability); neither server process swapped.

Short-workload measurements used three sequential repetitions and two concurrent pairs, compared with the **prior short-workload reliability baseline**, not the fresh long-context replay:

| Workload / server-reported rate | Full port | Reliability |
| --- | ---: | ---: |
| 2,293-token prefill | 942.51 tok/s | 907.00 tok/s |
| 9,169-token prefill | 978.26 tok/s | 935.66 tok/s |
| Decode after 61-token prompt | 58.74 tok/s | 56.39 tok/s |
| Decode after 2,293-token prompt | 58.39 tok/s | 51.81 tok/s |
| Decode after 9,169-token prompt | 61.39 tok/s | 61.43 tok/s |
| Two-request aggregate generation | 81.05 tok/s | 67.68 tok/s |

These measurements qualify this model and configuration, not every production workload or a general MTP-versus-DFlash2 comparison. Inherited upstream benchmark tables in other guides describe their original hardware and are not measurements of this fork.

## Lineage and maintenance

This branch preserves upstream history through `a754fcf3`, then adds the V100 platform, kernels, and optimizations. Future upstream updates can be merged normally, but CUDA/platform assumptions, changed Ops, artifact bindings, scheduler behavior, and serving contracts still need V100 validation. It is not a promise of conflict-free or automatic upstream compatibility.

Credit goes to [Neroued/NInfer](https://github.com/Neroued/ninfer), [geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100), and [liujun-7788/ninfer-v3-v100](https://github.com/liujun-7788/ninfer-v3-v100). Kernel and packing work also draws on Infernix and ninfer-all; exact provenance and retained third-party licensing are recorded in [the Volta notice](third_party/volta/NOTICE).

NInfer is licensed under [Apache 2.0](LICENSE), with the applicable third-party notices retained. See the [documentation index](docs/README.md) and [benchmark guide](bench/README.md) for further technical material.
