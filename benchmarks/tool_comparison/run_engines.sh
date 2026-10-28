#!/usr/bin/env bash
# Run the OCR engine x layout matrix of engines.config.json through docling-serve, then build its
# dashboard. Every configuration runs in a test image from Dockerfile.engines (our image plus the
# Egret layout weights, and Nemotron OCR v2 for its own configurations), one after another on DOCLING_GPU.
# GLM-OCR and the two full-page VLM configurations need a vLLM server: the script starts one per
# model on VLLM_GPU, in a docker network shared with docling-serve, and removes it afterwards.
#
#   benchmarks/tool_comparison/run_engines.sh [extra runner args, e.g. --force]
#   ONLY="nemotron-ocr rapidocr-egret-large" benchmarks/tool_comparison/run_engines.sh
set -euo pipefail
cd "$(dirname "$0")"

BASE_IMAGE="${BASE_IMAGE:-ghcr.io/dcc-bs/dcc-docling-serve-cu130:v1.38.1}"
IMAGE="${IMAGE:-dcc-docling-serve-test:v1.38.1-cu130-engines}"
# Only for the Nemotron OCR configurations: its torch 2.11 slows everything else down (Dockerfile.engines)
NEMOTRON_IMAGE="${NEMOTRON_IMAGE:-dcc-docling-serve-test:v1.38.1-cu130-nemotron}"
# The GLM-OCR configurations can run in an image with a patched plugin (the default is IMAGE)
GLM_IMAGE="${GLM_IMAGE:-$IMAGE}"
DOCLING_GPU="${DOCLING_GPU:-0}"
VLLM_GPU="${VLLM_GPU:-1}"
VLLM_MEMORY="${VLLM_MEMORY:-0.85}"  # --gpu-memory-utilization; lower it when the GPU is shared
HF_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}"
ONLY="${ONLY:-}"  # space-separated configuration ids; empty runs all of them
REPEATS="${REPEATS:-3}"  # conversions per document; the dashboard shows the median
ALLOW_BUSY_GPU="${ALLOW_BUSY_GPU:-0}"  # 1 starts even when other processes use the GPUs
NETWORK=tool-comparison-engines
RESULTS="${RESULTS:-results/engines}"
DOCS=(../../../test-docs synthetic ../../data/ocr.png public/xfund_de)  # public/: fetch_public.py

# Nemotron 3.5 Parse: the vLLM plugin ships inside the model repo and pins vllm 0.26.0
NEMOTRON_PARSE_REVISION=e7ea79242a87ea0ad3919b672bfedb429a683206
NEMOTRON_PARSE_DIR=/root/.cache/huggingface/nemotron-3.5-parse
LIGHTON_REVISION=b71010f095dc3735de7ee6b9969bff8f8f50b39a

LAYOUT_PRESETS='{
  "ppdoclayout-v3": {"kind": "ppdoclayout-v3"},
  "egret-medium": {"kind": "layout_object_detection", "model_spec": {"name": "layout_egret_medium", "repo_id": "docling-project/docling-layout-egret-medium", "revision": "main"}},
  "egret-large": {"kind": "layout_object_detection", "model_spec": {"name": "layout_egret_large", "repo_id": "docling-project/docling-layout-egret-large", "revision": "main"}},
  "egret-xlarge": {"kind": "layout_object_detection", "model_spec": {"name": "layout_egret_xlarge", "repo_id": "docling-project/docling-layout-egret-xlarge", "revision": "main"}}
}'
SERVER_ARGS=(
    --image "$IMAGE" --results "$RESULTS" --docs "${DOCS[@]}" --gpu-device "$DOCLING_GPU"
    --docker-arg="--network=$NETWORK"
    --env DOCLING_SERVE_ENABLE_REMOTE_SERVICES=true
    --env DOCLING_SERVE_ALLOW_EXTERNAL_PLUGINS=true
    --env "DOCLING_SERVE_CUSTOM_LAYOUT_PRESETS=$(jq -c . <<<"$LAYOUT_PRESETS")"
    --env "DOCLING_SERVE_CUSTOM_VLM_PRESETS=$(jq -c . vlm_presets.json)"
    --env PP_DOC_LAYOUT_CONFIDENCE_THRESHOLD=0.3
    --env GLMOCR_REMOTE_OCR_API_URL=http://engines-vllm-glm-ocr:8000/v1/chat/completions
)
# Untimed before every configuration: a scan, tables and handwriting load every stage's models
SERVER_ARGS+=(--repeats "$REPEATS" --warmup synthetic/synthetic_scan.pdf synthetic/synthetic_tables.pdf ../../data/ocr.png)
EXTRA_ARGS=("$@")

log() { echo "[engines] $*" >&2; }

wanted() { [[ -z "$ONLY" || " $ONLY " == *" $1 "* ]]; }

run_config() {  # <id> [--option KEY=VALUE ...]
    local id="$1"
    shift
    wanted "$id" || return 0
    log "=== $id"
    uv run --script run_docling.py --tool "$id" "${SERVER_ARGS[@]}" "$@" "${EXTRA_ARGS[@]}"
}

VLLM_CONTAINER=""
stop_vllm() {
    if [[ -n "$VLLM_CONTAINER" ]]; then
        docker rm -f "$VLLM_CONTAINER" >/dev/null 2>&1 || true
        VLLM_CONTAINER=""
    fi
}
SAMPLERS=()
cleanup() {
    stop_vllm
    (( ${#SAMPLERS[@]} )) && kill "${SAMPLERS[@]}" 2>/dev/null || true
}
trap cleanup EXIT

start_vllm() {  # <name> <image> <bash command run in the container>
    stop_vllm
    VLLM_CONTAINER="engines-vllm-$1"
    log "starting $VLLM_CONTAINER ($2) on GPU $VLLM_GPU"
    docker run -d --rm --name "$VLLM_CONTAINER" --network "$NETWORK" --gpus "device=$VLLM_GPU" --ipc=host \
        -v "$HF_CACHE:/root/.cache/huggingface" -e "HF_TOKEN=${HF_TOKEN:-}" \
        --entrypoint /bin/bash "$2" -c "$3" >/dev/null
    local elapsed=0
    until docker exec "$VLLM_CONTAINER" python3 -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/models', timeout=5)" 2>/dev/null; do
        if ! docker ps --format '{{.Names}}' | grep -qx "$VLLM_CONTAINER"; then
            log "$VLLM_CONTAINER exited; rerun with the container kept to read its logs"
            VLLM_CONTAINER=""
            return 1
        fi
        (( elapsed += 5 ))
        if (( elapsed > 900 )); then
            log "$VLLM_CONTAINER not ready after 900s: $(docker logs --tail 5 "$VLLM_CONTAINER" 2>&1)"
            return 1
        fi
        sleep 5
    done
    log "$VLLM_CONTAINER ready after ${elapsed}s"
}

any_wanted() { local id; for id in "$@"; do wanted "$id" && return 0; done; return 1; }

# --- Nothing else may run on the GPUs: other work changes the times ---------------------------
gpu_processes() {  # compute processes on GPU index $1
    local uuid
    uuid=$(nvidia-smi -i "$1" --query-gpu=uuid --format=csv,noheader)
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader | grep "^$uuid" || true
}
for gpu in $(printf '%s\n' "$DOCLING_GPU" "$VLLM_GPU" | sort -u); do
    busy=$(gpu_processes "$gpu")
    if [[ -n "$busy" && "$ALLOW_BUSY_GPU" != 1 ]]; then
        log "GPU $gpu is in use; stop these or set ALLOW_BUSY_GPU=1:"$'\n'"$busy"
        exit 1
    fi
done
[[ "$DOCLING_GPU" != "$VLLM_GPU" ]] || log "warning: docling-serve and vLLM share GPU $DOCLING_GPU; times differ from runs on two GPUs"
mkdir -p "$RESULTS"
{ date -Is; echo "load: $(cut -d' ' -f1-3 /proc/loadavg), cpus: $(nproc)"; docker ps --format '{{.Names}}\t{{.Image}}'; } \
    > "$RESULTS/_environment_$(date +%Y%m%d-%H%M%S).txt"
# GPU use and processes for the whole run, to find interference afterwards
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 5 >> "$RESULTS/_gpu_usage.csv" &
SAMPLERS+=($!)
nvidia-smi --query-compute-apps=timestamp,gpu_uuid,pid,process_name,used_memory --format=csv -l 30 >> "$RESULTS/_gpu_processes.csv" &
SAMPLERS+=($!)
awk '{print strftime("%FT%T"), $1, $2, $3}' /proc/loadavg >> "$RESULTS/_load.txt"
( while sleep 30; do awk '{print strftime("%FT%T"), $1, $2, $3}' /proc/loadavg; done >> "$RESULTS/_load.txt" ) &
SAMPLERS+=($!)

docker network inspect "$NETWORK" >/dev/null 2>&1 || docker network create "$NETWORK" >/dev/null
docker build -q -f Dockerfile.engines --build-arg BASE_IMAGE="$BASE_IMAGE" --target engines -t "$IMAGE" . >/dev/null
if any_wanted nemotron-ocr nemotron-ocr-pp nemotron-ocr-egret-large; then
    docker build -q -f Dockerfile.engines --build-arg BASE_IMAGE="$BASE_IMAGE" --target engines-nemotron \
        -t "$NEMOTRON_IMAGE" . >/dev/null
fi

# --- In-container engines (no vLLM) ---------------------------------------------------------
run_config release-rapidocr
run_config release-rapidocr-pp --option layout_preset=ppdoclayout-v3
for size in medium large xlarge; do
    run_config "rapidocr-egret-$size" --option "layout_preset=egret-$size"
done
# Nemotron validates no Latin language but English; iso:de goes to the English recognizer (best effort)
NEMOTRON_OCR=(--image "$NEMOTRON_IMAGE" --option ocr_preset=nemotron-ocr --option ocr_lang=iso:de)
run_config nemotron-ocr "${NEMOTRON_OCR[@]}"
run_config nemotron-ocr-pp "${NEMOTRON_OCR[@]}" --option layout_preset=ppdoclayout-v3
run_config nemotron-ocr-egret-large "${NEMOTRON_OCR[@]}" --option layout_preset=egret-large
PP_OCRV6=(--option ocr_preset=pp-ocrv6)
run_config pp-ocrv6 "${PP_OCRV6[@]}"
run_config pp-ocrv6-pp "${PP_OCRV6[@]}" --option layout_preset=ppdoclayout-v3
run_config pp-ocrv6-egret-large "${PP_OCRV6[@]}" --option layout_preset=egret-large

# --- GLM-OCR per region (as in compose.yaml) ------------------------------------------------
if any_wanted release-glm release-glm-pp glm-egret-large; then
    start_vllm glm-ocr vllm/vllm-openai:v0.30.0 \
        "exec vllm serve zai-org/GLM-OCR --served-model-name zai-org/GLM-OCR --port 8000 \
         --trust-remote-code --max-num-batched-tokens 8192 --max-model-len 32768 \
         --gpu-memory-utilization $VLLM_MEMORY"
    GLM=(--image "$GLM_IMAGE" --option ocr_preset=glm-ocr-remote)
    run_config release-glm "${GLM[@]}"
    run_config release-glm-pp "${GLM[@]}" --option layout_preset=ppdoclayout-v3
    run_config glm-egret-large "${GLM[@]}" --option layout_preset=egret-large
fi

# --- Full page through the VLM pipeline (presets in vlm_presets.json) -----------------------
if wanted vlm-nemotron-parse; then
    # Model-card flags minus --attention-config flash_attn_version 3, which needs a Hopper GPU
    start_vllm nemotron-parse vllm/vllm-openai:v0.26.0 \
        "python3 -c \"from huggingface_hub import snapshot_download; snapshot_download('nvidia/Nemotron-3.5-Parse', revision='$NEMOTRON_PARSE_REVISION', local_dir='$NEMOTRON_PARSE_DIR')\" \
         && uv pip install --system '$NEMOTRON_PARSE_DIR[vllm]' \
         && exec vllm serve nvidia/Nemotron-3.5-Parse --revision $NEMOTRON_PARSE_REVISION \
            --served-model-name nemotron-parse --port 8000 --trust-remote-code --dtype bfloat16 \
            --max-model-len 16384 --max-num-seqs 64 --max-num-batched-tokens 16384 \
            --no-enable-prefix-caching --limit-mm-per-prompt '{\"image\":1}' \
            --chat-template $NEMOTRON_PARSE_DIR/chat_template.jinja --gpu-memory-utilization $VLLM_MEMORY"
    run_config vlm-nemotron-parse --option pipeline=vlm --option vlm_pipeline_preset=nemotron-parse-3.5
fi
if wanted vlm-lighton-ocr-3; then
    start_vllm lighton-ocr-3 vllm/vllm-openai:v0.31.0 \
        "exec vllm serve lightonai/LightOnOCR-3-4B --revision $LIGHTON_REVISION \
         --served-model-name lightonai/LightOnOCR-3-4B --port 8000 --trust-remote-code \
         --max-model-len 16384 --limit-mm-per-prompt '{\"image\":1}' \
         --default-chat-template-kwargs '{\"enable_thinking\":false}' --gpu-memory-utilization $VLLM_MEMORY"
    # About 20x slower than the other configurations: one conversion per document
    run_config vlm-lighton-ocr-3 --option pipeline=vlm --option vlm_pipeline_preset=lighton-ocr-3 --repeats 1
fi
stop_vllm

uv run --script build_dashboard.py --config engines.config.json --results "$RESULTS"
