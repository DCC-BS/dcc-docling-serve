"""Compare the markdown two runners produced for the same documents.

Used in the upgrade checklist: run docling with and without the shape patch, then check
that the markdown is still (almost) the same.

    python benchmarks/tool_comparison/compare_outputs.py docling-gpu-ocr docling-gpu-ocr-noshape
"""

import argparse
import difflib
import json
from pathlib import Path

from common import DEFAULT_RESULTS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--min-similarity", type=float, default=0.97, help="Flag documents below this (default 0.97)")
    args = parser.parse_args()

    left_dir, right_dir = args.results / "outputs" / args.left, args.results / "outputs" / args.right
    identical, flagged, total = 0, [], 0
    for left_file in sorted(left_dir.glob("*.json")):
        right_file = right_dir / left_file.name
        if left_file.name == "_setup.json" or not right_file.exists():
            continue
        left, right = json.loads(left_file.read_text()), json.loads(right_file.read_text())
        total += 1
        if left["markdown"] == right["markdown"]:
            identical += 1
            continue
        a, b = left["markdown"], right["markdown"]
        ratio = difflib.SequenceMatcher(None, a, b, autojunk=len(a) + len(b) > 400_000).ratio()
        marker = "  <-- check" if ratio < args.min_similarity else ""
        print(f"{left['doc']:60} similarity {ratio:.4f}  ({len(a)} vs {len(b)} chars){marker}")
        if ratio < args.min_similarity:
            flagged.append(left["doc"])
    print(f"\n{identical}/{total} documents identical, {len(flagged)} below {args.min_similarity}")
    if flagged:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
