#!/usr/bin/env bash
# Production launcher for the full upstream Volta engine, validated at f4e186aa.
set -euo pipefail
model_path="${MODEL_PATH:-/models/qwen3.8-27b-volta-v3.ninfer}"
test -r "$model_path" || { echo "Model is not readable: $model_path" >&2; exit 1; }
args=(
    "$model_path"
    --host 0.0.0.0 --port "${PORT:-8080}" --device "${DEVICE:-0}"
    --model-id "${MODEL_ID:-qwen3.8-27b}"
    --max-context "${MAX_CONTEXT:-232000}" --context-capacity auto
    --kv-dtype "${KV_DTYPE:-int8}" --prefill-chunk "${PREFILL_CHUNK:-2048}"
    --max-concurrency "${MAX_CONCURRENCY:-2}"
    --host-cache-capacity "${HOST_CACHE_CAPACITY:-9221505024B}"
    --prefix-reuse --cuda-graph
    --spec mtp --draft-tokens "${DRAFT_TOKENS:-4}" --proposal-head optimized
    --preserve-thinking --pending-timeout "${PENDING_TIMEOUT:-600s}"
    --gpu-timing --log-level "${LOG_LEVEL:-info}"
)
if [[ "${VISION:-1}" == 1 ]]; then args+=(--vision); fi
if [[ -n "${REQUEST_LOG_JSONL:-}" ]]; then args+=(--request-log-jsonl "$REQUEST_LOG_JSONL"); fi
if [[ -n "${API_KEY:-}" ]]; then args+=(--api-key "$API_KEY"); fi
exec /opt/ninfer/bin/ninfer-serve "${args[@]}"
