# /// script
# requires-python = ">=3.12"
# dependencies = ["markitdown[pdf]"]
# ///
"""Convert the documents with Microsoft markitdown (PDF text extraction, no OCR, no LLM).

uv run --script run_markitdown.py
"""

import importlib.metadata
import time

from common import Recorder, list_docs, parser


def main() -> None:
    args = parser(__doc__).parse_args()
    recorder = Recorder("markitdown", args.results, args.force)
    docs = list_docs(args.docs, args.only)
    t0 = time.perf_counter()
    from markitdown import MarkItDown

    converter = MarkItDown()
    recorder.setup(model_load_s=time.perf_counter() - t0, version=importlib.metadata.version("markitdown"))
    recorder.run(docs, lambda doc: {"markdown": converter.convert(str(doc)).text_content, "pages": None, "meta": {}})


if __name__ == "__main__":
    main()
