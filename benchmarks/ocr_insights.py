# /// script
# requires-python = ">=3.12"
# dependencies = ["docling[rapidocr]"]
# ///
"""Which pages and layout regions docling sends to OCR, and why.

Without force_ocr, docling's default OCR mode (pdf_aware_layout_regions) OCRs every
layout cluster that either
  - overlaps a bitmap or a vector shape (lines, borders, filled boxes), even when it
    also contains programmatic PDF text, or
  - contains no programmatic PDF text at all.
This script converts the documents with the benchmark pipeline (see local_docling.py)
and records, per page, each cluster's reason, the resulting OCR calls, the rendered
crop pixels and the OCR engine time.

    uv run --script benchmarks/ocr_insights.py ../test-docs
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import rapidocr
from docling.datamodel.base_models import Page
from docling.datamodel.settings import settings
from docling.models.base_ocr_model import BaseOcrModel
from local_docling import build_converter

OPTIONS = {"do_ocr": True, "table_mode": "accurate", "ocr_lang": ["de"]}
SUPPORTED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

# Filled while converting: document -> page number -> stats
STATS: dict[str, dict[int, dict]] = defaultdict(dict)
CURRENT = {"doc": "", "page": None}


def cluster_reason(backend, bbox) -> str:
    """Why the default OCR mode keeps (or drops) a layout cluster."""
    bitmap = backend.has_content_in(bbox=bbox, chars=False, shapes=False, bitmaps=True)
    if bitmap is None:
        return "unknown (backend has no content queries)"
    if bitmap:
        return "bitmap"
    if backend.has_content_in(bbox=bbox, chars=False, shapes=True, bitmaps=False):
        has_text = backend.has_content_in(bbox=bbox, chars=True, shapes=False, bitmaps=False)
        return "shape + pdf text" if has_text else "shape, no pdf text"
    if not backend.has_content_in(bbox=bbox, chars=True, shapes=False, bitmaps=False):
        return "no pdf text"
    return "skipped (pdf text only)"


def instrument() -> None:
    original_get_ocr_rects = BaseOcrModel.get_ocr_rects

    def get_ocr_rects(self, page: Page):
        rects = original_get_ocr_rects(self, page)
        reasons: Counter = Counter()
        labels: Counter = Counter()
        if page.predictions.layout is not None and page._backend is not None:
            for cluster in page.predictions.layout.clusters:
                reason = cluster_reason(page._backend, cluster.bbox)
                reasons[reason] += 1
                if not reason.startswith("skipped"):
                    labels[f"{cluster.label.value}: {reason}"] += 1
        page_area = page.size.width * page.size.height if page.size else 0
        STATS[CURRENT["doc"]][page.page_no] = {
            "clusters": sum(reasons.values()),
            "reasons": dict(reasons),
            "labels": dict(labels),
            "ocr_rects": len(rects),
            "ocr_area_fraction": sum(r.area() for r in rects) / page_area if page_area else 0,
            "ocr_calls": 0,
            "ocr_pixels": 0,
            "ocr_seconds": 0.0,
            "ocr_lines": 0,
        }
        CURRENT["page"] = page.page_no
        return rects

    BaseOcrModel.get_ocr_rects = get_ocr_rects

    original_call = rapidocr.RapidOCR.__call__

    def call(self, image, *args, **kwargs):
        t0 = time.perf_counter()
        result = original_call(self, image, *args, **kwargs)
        page = STATS[CURRENT["doc"]].get(CURRENT["page"])
        if page is not None:
            height, width = image.shape[:2] if hasattr(image, "shape") else (image.height, image.width)
            page["ocr_calls"] += 1
            page["ocr_pixels"] += width * height
            page["ocr_seconds"] += time.perf_counter() - t0
            page["ocr_lines"] += len(getattr(result, "txts", None) or [])
        return result

    rapidocr.RapidOCR.__call__ = call


def report(docs: list[Path]) -> str:
    lines = ["# OCR insights", "", f"Options: `{json.dumps(OPTIONS)}` (force_ocr off, OCR mode default)", ""]
    lines += ["## Per document", ""]
    lines += ["| Document | Pages | Pages with OCR | Clusters | OCR calls | OCR area % | OCR Mpx | OCR s | Lines |"]
    lines += ["|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    totals: Counter = Counter()
    for doc in docs:
        pages = STATS.get(doc.name, {})
        calls = sum(p["ocr_calls"] for p in pages.values())
        area = sum(p["ocr_area_fraction"] for p in pages.values()) / max(1, len(pages))
        seconds = sum(p["ocr_seconds"] for p in pages.values())
        pixels = sum(p["ocr_pixels"] for p in pages.values())
        lines.append(
            f"| {doc.name} | {len(pages)} | {sum(1 for p in pages.values() if p['ocr_calls'])} "
            f"| {sum(p['clusters'] for p in pages.values())} | {calls} | {area * 100:.0f} "
            f"| {pixels / 1e6:.0f} | {seconds:.1f} | {sum(p['ocr_lines'] for p in pages.values())} |"
        )
        for p in pages.values():
            totals.update({f"reason: {k}": v for k, v in p["reasons"].items()})
            totals.update({f"label: {k}": v for k, v in p["labels"].items()})
            totals["ocr_seconds"] += p["ocr_seconds"]

    lines += ["", "## Why clusters went to OCR (all documents)", "", "| Reason | Clusters |", "|---|---:|"]
    for key, count in sorted(((k, v) for k, v in totals.items() if k.startswith("reason: ")), key=lambda kv: -kv[1]):
        lines.append(f"| {key.removeprefix('reason: ')} | {count} |")

    lines += ["", "## OCR'd clusters by layout label and reason", "", "| Label: reason | Clusters |", "|---|---:|"]
    for key, count in sorted(((k, v) for k, v in totals.items() if k.startswith("label: ")), key=lambda kv: -kv[1]):
        lines.append(f"| {key.removeprefix('label: ')} | {count} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("docs", type=Path, help="Folder with the documents to analyse")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results" / "ocr_insights")
    args = parser.parse_args()

    docs = sorted(p.resolve() for p in args.docs.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)
    settings.debug.profile_pipeline_timings = True
    instrument()
    converter = build_converter(OPTIONS)
    for doc in docs:
        print(f"converting {doc.name}", file=sys.stderr, flush=True)
        CURRENT.update(doc=doc.name, page=None)
        converter.convert(doc, raises_on_error=False)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pages.json").write_text(json.dumps(STATS, indent=2))
    summary = report(docs)
    (args.out / "summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
