#!/usr/bin/env bash
# Run every tool over the test documents and the synthetic set, then build the dashboard.
# Local tools run one after another on the same GPU so they do not compete for CPU/GPU;
# Jina runs in parallel because its work happens on Jina's servers.
#
#   benchmarks/tool_comparison/run_all.sh [extra runner args, e.g. --force]
set -euo pipefail
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" TORCH_DEVICE=cuda
PATCHED_IMAGE="${PATCHED_IMAGE:-dcc-docling-serve-test:v1.35.0-cu130}"
OFFICIAL_IMAGE="${OFFICIAL_IMAGE:-ghcr.io/docling-project/docling-serve-cu130:v1.35.0}"

uv run --script make_synthetic.py

uv run --script run_jina.py "$@" &
JINA_PID=$!

uv run --script run_docling.py --tool docling-today --image "$OFFICIAL_IMAGE" "$@"
uv run --script run_docling.py --tool docling-gpu-ocr --image "$PATCHED_IMAGE" --env DCC_OCR_IGNORE_SHAPES=0 "$@"
uv run --script run_docling.py --tool docling-gpu-ocr-noshape --image "$PATCHED_IMAGE" --env DCC_OCR_IGNORE_SHAPES=1 "$@"
uv run --script run_marker.py "$@"
uv run --script run_doctr.py "$@"
uv run --script run_markitdown.py "$@"

wait "$JINA_PID" || echo "jina runner failed" >&2
uv run --script build_dashboard.py
