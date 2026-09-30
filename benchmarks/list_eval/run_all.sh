#!/usr/bin/env bash
# List detection benchmark: Heron vs PP-DocLayout-V3 with list_detection off / rules / heron.
#   benchmarks/list_eval/run_all.sh <image> [results dir]
# The image needs docling-pp-doc-layout >= 0.2.6 (e.g. the published cu130 image or Dockerfile.debug).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
IMAGE="${1:?image}"
RESULTS="${2:-$HERE/results}"
TEST_DOCS="${TEST_DOCS:-$HERE/../../../test-docs}"
DOCS=(/e/lists_styles.pdf /e/lists_styles_scan.pdf /e/lists_columns.pdf /e/lists_columns_scan.pdf)
for f in "$TEST_DOCS"/*; do
  case "$f" in *.pdf|*.jpg|*.png) DOCS+=("/t/$(basename "$f")") ;; esac
done
mkdir -p "$RESULTS"
for cfg in "heron:heron:off" "pp-off:pp:off" "pp-rules:pp:rules" "pp-heron:pp:heron"; do
  IFS=: read -r name layout lists <<< "$cfg"
  echo "== $name"
  docker run --rm --gpus device=0 --entrypoint python -e LAYOUT="$layout" -e PP_DOC_LAYOUT_LIST_DETECTION="$lists" \
    -v "$HERE/docs:/e:ro" -v "$TEST_DOCS:/t:ro" -v "$HERE/convert.py:/convert.py:ro" \
    -v "$RESULTS:/out" "$IMAGE" /convert.py "/out/$name" "${DOCS[@]}"
done
uv run --script "$HERE/evaluate.py" "$RESULTS"
