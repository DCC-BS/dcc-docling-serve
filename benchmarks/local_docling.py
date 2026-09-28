# /// script
# requires-python = ">=3.12"
# dependencies = ["docling[rapidocr]"]
# ///
"""Time plain (non-Docker) docling with the pipeline docling-serve builds for the benchmark options.

Run by e2e_timing.py through ``uv run --script`` so it always gets the latest docling
from PyPI. Emits one JSON object per line on stdout; everything else goes to stderr.
"""

import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path

from docling.backend.docling_parse_backend import ThreadedDoclingParseDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    RapidOcrOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption
from docling_core.types.doc import ImageRefMode


def emit(record: dict) -> None:
    print(json.dumps(record), flush=True)


def build_converter(options: dict) -> DocumentConverter:
    # Mirrors docling_jobkit's DoclingConverterManager.get_pdf_pipeline_opts for
    # {do_ocr, pdf_backend=docling_parse, table_mode, ocr_preset=rapidocr, ocr_lang}
    # plus docling-serve's defaults (include_images=true, images_scale=2).
    pipeline_options = PdfPipelineOptions(
        do_ocr=options.get("do_ocr", True),
        ocr_options=RapidOcrOptions(lang=options.get("ocr_lang", ["de"])),
        do_table_structure=True,
        table_structure_options=TableStructureOptions(mode=TableFormerMode(options.get("table_mode", "accurate"))),
        generate_picture_images=True,
        images_scale=2.0,
    )
    format_option = {"pipeline_options": pipeline_options, "backend": ThreadedDoclingParseDocumentBackend}
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(**format_option),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
        }
    )


def environment() -> dict:
    info = {}
    for package in ("docling", "docling-slim", "docling-core", "rapidocr", "onnxruntime", "onnxruntime-gpu", "torch"):
        try:
            info[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        import onnxruntime

        info["onnxruntime_providers"] = onnxruntime.get_available_providers()
    except ImportError:
        pass
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--options", required=True, help="Benchmark options as JSON")
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--warmup", type=Path, required=True, help="Document converted once before timing")
    parser.add_argument("docs", nargs="+", type=Path)
    args = parser.parse_args()

    settings.debug.profile_pipeline_timings = True
    emit({"kind": "environment", **environment()})

    converter = build_converter(json.loads(args.options))

    def convert(path: Path) -> dict:
        t0 = time.perf_counter()
        result = converter.convert(path, raises_on_error=False)
        # docling-serve exports markdown with embedded images by default, so do the same.
        result.document.export_to_markdown(image_mode=ImageRefMode.EMBEDDED)
        wall = time.perf_counter() - t0
        timings = {
            name: {"scope": str(item.scope.value), "count": item.count, "total": sum(item.times)}
            for name, item in (result.timings or {}).items()
        }
        return {"status": result.status.value, "wall_s": wall, "timings": timings}

    # The pipeline and its models are created lazily, so the first conversion is the cold start.
    emit({"kind": "cold", "doc": args.warmup.name, **convert(args.warmup)})

    for repeat in range(args.repeats):
        for doc in args.docs:
            emit({"kind": "run", "doc": doc.name, "repeat": repeat, **convert(doc)})
            print(f"local: {doc.name} #{repeat + 1} done", file=sys.stderr)


if __name__ == "__main__":
    main()
