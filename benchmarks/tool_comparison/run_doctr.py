# /// script
# requires-python = ">=3.12"
# dependencies = ["python-doctr", "torch", "torchvision", "huggingface_hub"]
# ///
"""Convert the documents with docTR: pure OCR of rendered pages (it ignores the PDF text layer).

Uses the multilingual PARSeq recognizer from the Hugging Face hub (the built-in models only
know the French vocabulary, without ä/ö/ü/ß) plus docTR's layout and table detection so
export_as_markdown produces headings and tables.

    CUDA_VISIBLE_DEVICES=0 uv run --script run_doctr.py
"""

import importlib.metadata
import time

from common import Recorder, list_docs, parser

RECOGNIZER = "Felix92/doctr-torch-parseq-multilingual-v1"
PAGES_PER_BATCH = 16  # bounds host and GPU memory on long documents


def main() -> None:
    args = parser(__doc__).parse_args()
    recorder = Recorder("doctr", args.results, args.force)
    docs = list_docs(args.docs, args.only)

    t0 = time.perf_counter()
    import pypdfium2
    from doctr.io import DocumentFile
    from doctr.models import from_hub, ocr_predictor

    model = ocr_predictor(
        det_arch="fast_base",
        reco_arch=from_hub(RECOGNIZER),
        pretrained=True,
        detect_layout=True,
        detect_tables=True,
    ).cuda()
    load_s = time.perf_counter() - t0

    def convert(doc):
        pages_md: list[str] = []
        if doc.suffix.lower() == ".pdf":
            count = len(pypdfium2.PdfDocument(doc))
            for start in range(0, count, PAGES_PER_BATCH):
                # from_pdf renders at 144 dpi (scale 2), docTR's default
                pdf = pypdfium2.PdfDocument(doc)
                images = [
                    pdf[i].render(scale=2).to_numpy()[..., :3]
                    for i in range(start, min(start + PAGES_PER_BATCH, count))
                ]
                pdf.close()
                pages_md += [page.export_as_markdown() for page in model(images).pages]
        else:
            pages_md += [page.export_as_markdown() for page in model(DocumentFile.from_images(str(doc))).pages]
        return {"markdown": "\n\n".join(pages_md), "pages": pages_md, "meta": {"recognizer": RECOGNIZER}}

    warm_t0 = time.perf_counter()
    convert(docs[0])
    recorder.setup(model_load_s=load_s, first_conversion_s=time.perf_counter() - warm_t0,
                   version=importlib.metadata.version("python-doctr"), recognizer=RECOGNIZER)  # fmt: skip
    recorder.run(docs, convert)


if __name__ == "__main__":
    main()
