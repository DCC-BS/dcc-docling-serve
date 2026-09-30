"""Convert documents with docling + PP-DocLayout-V3 (or Heron) and save the item labels.

Runs inside a dcc-docling-serve image (see run_all.sh). LAYOUT=heron|pp selects the layout
model; PP_DOC_LAYOUT_LIST_DETECTION selects the plugin's list detection.

    python convert.py <out dir> <documents...>
"""

import json
import os
import sys
import time
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption
from docling_pp_doc_layout.options import PPDocLayoutV3Options

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
options = PdfPipelineOptions(do_ocr=True, ocr_options=RapidOcrOptions(lang=["de"]), allow_external_plugins=True)
if os.environ.get("LAYOUT", "pp") == "pp":
    options.layout_options = PPDocLayoutV3Options()
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=options),
        InputFormat.IMAGE: ImageFormatOption(pipeline_options=options),
    }
)
converter.initialize_pipeline(InputFormat.PDF)
for path in sys.argv[2:]:
    t0 = time.perf_counter()
    doc = converter.convert(path).document
    seconds = time.perf_counter() - t0
    items = [
        (item.label.value, item.text, getattr(item, "marker", None))
        for item, _ in doc.iterate_items(with_groups=False)
        if hasattr(item, "text")
    ]
    record = {"doc": Path(path).name, "seconds": seconds, "items": items, "md": doc.export_to_markdown()}
    (out / f"{Path(path).name}.json").write_text(json.dumps(record, ensure_ascii=False))
    n = sum(1 for label, *_ in items if label == "list_item")
    print(f"{Path(path).name}: {seconds:.1f}s, {n} list items", flush=True)
