# /// script
# requires-python = ">=3.12"
# dependencies = ["lightonocr @ git+https://github.com/lightonai/LightOnOCR@36755d461be079737860a5f03ae0c803501269e9"]
# ///
"""Convert the documents with LightOnOCR-3 on its own vLLM server, without docling.

Uses LightOn's client as its CLI does: every page of a document rendered at the model's page size
(2048 px longest edge), all pages sent at once (one image per request, vLLM batches them), the
CLI's sampling (temperature 0.2, no token cap). In grounding mode the layout blocks are joined back
into markdown without headers, footers and page numbers, as docling leaves them out too;
footnotes stay, as in docling.
run_engines.sh starts the server and runs this script.

    uv run --script run_lighton_ocr.py --url http://127.0.0.1:18011/v1 --tool lighton-ocr-3-vllm
"""

import statistics
import time

from common import Recorder, list_docs, log, parser
from lightonocr import DEFAULT_CONCURRENCY, HEADER_FOOTER, LightOnOCR, load_pages, ocr_pages, parse_blocks, to_markdown

# LightOn's HEADER_FOOTER also drops footnotes (olmOCR-bench counts them as footer content).
# Docling keeps footnotes in its markdown, and the PDF text layers we score against have them.
DROP = tuple(label for label in HEADER_FOOTER if "footnote" not in label)


def main() -> None:
    cli = parser(__doc__)
    cli.add_argument("--url", default="http://127.0.0.1:18011/v1", help="vLLM server, OpenAI base URL")
    cli.add_argument("--model", default="lightonai/LightOnOCR-3-4B", help="Served model name")
    cli.add_argument("--tool", default="lighton-ocr-3-vllm", help="Name of the results folder")
    cli.add_argument("--mode", choices=["grounding", "plain"], default="grounding", help="LightOnOCR prompt mode")
    cli.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Pages in flight at once")
    cli.add_argument("--repeats", type=int, default=1, help="Conversions per document; the median time is kept")
    args = cli.parse_args()
    ocr = LightOnOCR(base_url=args.url, model=args.model)
    recorder = Recorder(args.tool, args.results, args.force)
    recorder.setup(model_load_s=0.0, endpoint=args.url, model=args.model, mode=args.mode, concurrency=args.concurrency)
    # ocr_pages writes each page's image and raw output; kept outside outputs/, which the dashboard reads
    pages_dir = args.results / "pages" / args.tool

    def convert_once(doc) -> dict:
        t_start = time.perf_counter()  # includes rendering, as the docling times do
        images = load_pages(doc, ocr.profile.longest_edge)
        numbers = list(range(1, len(images) + 1))
        results = list(ocr_pages(ocr, pages_dir / doc.name, numbers, images, args.mode, concurrency=args.concurrency))
        seconds = time.perf_counter() - t_start
        if args.mode == "grounding":
            pages = [to_markdown(parse_blocks(r["raw"]), drop=DROP) for r in results]
        else:
            pages = [r["raw"] for r in results]
        empty = [r["page"] for r in results if not r["raw"].strip()]
        if empty:
            log(f"{doc.name}: no output on page(s) {empty}")
        return {
            "markdown": "\n\n".join(pages),
            "pages": pages,
            "seconds": seconds,
            "meta": {
                "page_images": len(images),
                "empty_pages": empty,
                "raw": "\n\n".join(r["raw"] for r in results)[:20000],
            },
        }

    def convert(doc) -> dict:
        runs = [convert_once(doc) for _ in range(args.repeats)]
        out = runs[0]
        out["meta"]["wall_runs_s"] = [r["seconds"] for r in runs]
        out["seconds"] = statistics.median(r["seconds"] for r in runs)
        return out

    recorder.run(list_docs(args.docs, args.only), convert)


if __name__ == "__main__":
    main()
