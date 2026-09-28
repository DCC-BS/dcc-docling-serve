# /// script
# requires-python = ">=3.12"
# dependencies = ["marker-pdf"]
# ///
"""Convert the documents with marker (surya models on the GPU, no LLM).

CUDA_VISIBLE_DEVICES=0 TORCH_DEVICE=cuda uv run --script run_marker.py
"""

import importlib.metadata
import re
import time

from common import Recorder, list_docs, parser

PAGE_SEPARATOR = re.compile(r"\n*\{\d+\}-{40,}\n*")


def main() -> None:
    args = parser(__doc__).parse_args()
    recorder = Recorder("marker", args.results, args.force)
    docs = list_docs(args.docs, args.only)

    t0 = time.perf_counter()
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered

    converter = PdfConverter(
        artifact_dict=create_model_dict(),
        config={"paginate_output": True, "disable_image_extraction": True, "output_format": "markdown"},
    )
    load_s = time.perf_counter() - t0

    def convert(doc):
        text, _, _ = text_from_rendered(converter(str(doc)))
        pages = [page.strip() for page in PAGE_SEPARATOR.split(text)]
        pages = pages[1:] if pages and not pages[0] else pages
        return {"markdown": "\n\n".join(pages), "pages": pages, "meta": {}}

    warm_t0 = time.perf_counter()
    convert(docs[0])
    recorder.setup(model_load_s=load_s, first_conversion_s=time.perf_counter() - warm_t0,
                   version=importlib.metadata.version("marker-pdf"))  # fmt: skip
    recorder.run(docs, convert)


if __name__ == "__main__":
    main()
