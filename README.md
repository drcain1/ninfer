# NInfer Volta — reliability build

A conservative update to the working Tesla V100 C++/CUDA inference engine, with selected upstream reliability fixes and measured performance parity with the previously deployed binary.

## Why this fork exists

The existing V100 engine worked well on our hardware, but its source had fallen hundreds of commits behind [Neroued/NInfer](https://github.com/Neroued/ninfer). Upstream had replaced substantial parts of the engine and targeted RTX 5090 and DGX Spark. Pulling all of that into the older V100 fork required a separate architecture port.

We kept two tracks in **[drcain1/ninfer-volta](https://github.com/drcain1/ninfer-volta)**:

| Branch | Purpose | Tested inference revision |
| --- | --- | --- |
| [`backport/upstream-reliability-20261103`](https://github.com/drcain1/ninfer-volta/tree/backport/upstream-reliability-20261103) — this branch | Keep the working V100 engine and backport selected reliability improvements | `bc3f1be2` |
| [`port/upstream-v100-20261103`](https://github.com/drcain1/ninfer-volta/tree/port/upstream-v100-20261103) | Port the newer upstream engine to Volta, including native kernels and performance work | `f4e186aa`, based on upstream `a754fcf3` |

This branch retains the older engine architecture. The full port has its own build presets, artifact requirements, and runtime flags. On November 5, 2026, the qualified full port replaced this reliability build on the reference host. This build and its deployment snapshot remain available for rollback. Documentation commits after the revisions above do not change the tested binaries.

## What changed

- **Transactional workspace allocation:** a failed allocation, size overflow, or alignment failure leaves the previous workspace layout intact rather than partially modifying it.
- **Linux listening-socket reuse:** `SO_REUSEADDR` permits immediate restart on the same port while preserving active-listener exclusivity. This does not enable `SO_REUSEPORT`.
- **Thinking-token accounting:** reasoning tokens are counted independently of the thinking budget and raw/presented response form.
- **Context-capacity diagnostics:** rejection messages report the configured limit and the prepared prompt length, distinguishing an exact count from a lower bound.

These fixes were backported from upstream without replacing the working V100 engine or changing the model weights. The serving products remain the CLI and OpenAI/Anthropic-compatible server using `.ninfer` artifacts.

## Target and build

The measured target is **Tesla V100 PCIe 32 GB (`sm_70`) on Linux x86_64**, with CUDA **12.8.1**. The reference deployment uses rootless Podman on AlmaLinux. Other Volta cards, V100 16 GB, and the full inherited model list have not been qualified by the tests below.

Use CMake 3.28+, Ninja, a compatible C++20 compiler, CUDA 12.8, and the development dependencies described in the inherited [V100 build guide](docs/V100-BUILD.md). That guide records the earlier maintainer's environment; its performance claims and high-context examples are historical, not results from this reliability campaign. The CUDA toolkit must retain `sm_70` support; CUDA 13 does not.

```bash
git clone --branch backport/upstream-reliability-20261103 https://github.com/drcain1/ninfer-volta.git
cd ninfer-volta
cmake -S . -B build-v100 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build build-v100 --parallel 4
```

Adjust the CUDA path for your installation. The build uses CUTLASS; the guide explains how to supply an existing checkout when needed. Products are under `build-v100/apps/`; preserve accompanying libraries when packaging.

## Validated serving profile

Validation used **Swift 1.5 Qwen3.8 27B OrcaRouter NVFP4**, served under the alias `qwen3.8-27b`. Its artifact includes DFlash2 weights, but these measurements use **MTP with four draft tokens** and the draft LM head.

```bash
./build-v100/apps/ninfer-serve /path/to/Swift-1.5-Qwen3.8-27B-OrcaRouter-NVFP4-DFlash2.ninfer \
  --host 127.0.0.1 --port 8080 --model-id qwen3.8-27b \
  --max-context 200000 --prefill-chunk 2048 --kv-capacity auto \
  --max-concurrency 2 --kv-dtype int8 \
  --device-state-slots 1 --host-state-slots 32 --host-kv-mib 4096 \
  --max-private-continuations 4 --max-shared-prefixes 0 \
  --spec mtp --draft-tokens 4 --lm-head-draft --preserve-thinking --vision
```

The tested profile uses CUDA graphs and private conversation caching. One extra device-state slot and 32 host-state slots accompany 4 GiB of host KV storage; that is not a cap on total process memory. This recipe uses the older engine's flags. Use the full-port branch's README when configuring that engine.

The [Podman deployment snapshot](docker/podman/README.txt) records the installed image `localhost/ninfer-v100:sm70-bc3f1be2`, launcher, Quadlet, and rollback location. The image and model are local deployment artifacts, not published container images or binaries included in this repository.

## Tests and measured results

The [November 4 report](docker/podman/validation/2026-11-04/REPORT.txt), [aggregate results](docker/podman/validation/2026-11-04/summary.json), and [samples](docker/podman/validation/2026-11-04/samples.csv) compare the actual deployed `sm70-v2` executable with the `bc3f1be2` replacement:

| Check | Result / coverage |
| --- | --- |
| Targeted CTest suites | 18/18 passed, no skips |
| Real-model HTTP regression | 16/16 passed on both old and replacement builds |
| Concurrent arithmetic | 8/8 passed on both builds |
| Sequential benchmark outputs | 12/12 exactly matched |
| Deployment checks after rollout | Health, chat, streaming, concurrency, thinking accounting, and executable/image identity passed |

HTTP checks covered arithmetic, JSON instructions, multi-turn history, thinking accounting, stream/nonstream parity, prefix reuse, stop strings, output limits, tool calls, Responses storage/retrieval/deletion, Anthropic messages/token counting, a simple image fixture, malformed requests, unknown models, context rejection, and recovery after a client disconnect.

The benchmark ran old/new/new/old in independent processes with warmup, the same model and sampling settings, no timed prompt-cache hits, a 170 W power limit, and dynamic GPU clocks. Startup time was excluded.

| Workload | Original | Reliability |
| --- | ---: | ---: |
| 61 input → 256 output tokens, median completion time | 4.949 s | 4.955 s |
| 2,293 input → 256 output tokens, median completion time | 7.304 s | 7.316 s |
| 9,169 input → 128 output tokens, median completion time | 11.943 s | 11.935 s |
| Two concurrent requests, aggregate generation | 67.93 tok/s | 67.83 tok/s |

Median completion times differed by less than 0.2%. These results show no meaningful speed change in the tested workloads. The longest timed input in that campaign was 9,169 tokens; the 200K boundary check tested rejection, not successful generation at 200K. The baseline executable's exact source/compiler configuration was not independently known. Matching text is functional regression evidence, not an independent numerical-kernel proof, and concurrent free-form text was not universally identical even across runs of the original binary.

The report records the pre-rollout state; the replacement was subsequently deployed, as described in the deployment snapshot.

## Later long-context comparison

On November 5, the reliability build also served as the baseline for the [full port's long-context qualification](https://github.com/drcain1/ninfer-volta/blob/port/upstream-v100-20261103/README.md#performance-and-host-memory). Each build completed 49 requests with 21 main continuations, reaching a final prompt of 180,210 tokens. Both retained advancing prefix reuse without cache-loss multi-minute warm stalls in that replay.

For reliability, cold 150K time to first token was **338.918 seconds**, warm median **7.223 seconds**, peak process RSS **10.142 GiB**, and minimum host memory available **3.732 GiB**, with no process swap. See the linked comparison for the full port's results and the different cache budgets.

Cold 150K processing still takes several minutes. Neither campaign qualifies 232K+ contexts, every concurrent workload, or every inherited model/kernel path. Vision coverage was a simple fixture, not a broad multimodal evaluation.

## Credits and further documentation

This work builds on [Neroued/NInfer](https://github.com/Neroued/ninfer), [geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100), and [liujun-7788/ninfer-v3-v100](https://github.com/liujun-7788/ninfer-v3-v100). The separate full-port branch records its additional kernel provenance and validation.

See the [documentation index](docs/README.md), [serving guide](docs/serving.md), and [test guide](tests/README.md). Inherited guides and benchmark tables describe their original hardware and scope unless explicitly qualified above. Licensed under [Apache 2.0](LICENSE), with applicable third-party notices retained.
