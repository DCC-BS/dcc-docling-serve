# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx", "pypdfium2", "pillow"]
# ///
"""Convert the documents with Unlimited-OCR-2 on its own SGLang server, without docling.

Follows the model card: a document goes out in one request with all its pages (each page one
global view, "base mode"); a single page is tiled ("gundam mode"). The answer's <PAGE> markers
split it into pages, and the grounding markers (block label and box) are removed, so what is
scored is the model's own markdown. run_engines.sh starts the server and runs this script.

    uv run --script run_unlimited_ocr.py --url http://127.0.0.1:18000 --tool unlimited-ocr-2-sglang
"""

import base64
import io
import os
import re
import statistics
import time
from pathlib import Path

import httpx
import pypdfium2
from common import Recorder, list_docs, parser
from PIL import Image

DPI = 300  # the model card rasterises PDFs at 300 DPI
TOKENS_PER_PAGE = 8000
PAGE_TOKEN_ID = 128827  # <PAGE>
# Grounding markers: <|det|>label [x1, y1, x2, y2]<|/det|> (v2) and <|ref|>label<|/ref|><|det|>[[..]]<|/det|> (v1)
_GROUNDING = re.compile(r"<\|ref\|>.*?<\|/ref\|>|<\|det\|>.*?<\|/det\|>", re.DOTALL)
_SPECIAL = re.compile(r"<｜[^｜]*｜>|<\|[^|]*\|>")


def page_images(doc: Path) -> list[bytes]:
    """PNG bytes per page: PDFs rendered at DPI, images as they are."""
    if doc.suffix.lower() != ".pdf":
        with Image.open(doc) as image, io.BytesIO() as buffer:
            image.convert("RGB").save(buffer, format="PNG")
            return [buffer.getvalue()]
    pdf = pypdfium2.PdfDocument(doc)
    try:
        pngs = []
        for page in pdf:
            with io.BytesIO() as buffer:
                page.render(scale=DPI / 72).to_pil().convert("RGB").save(buffer, format="PNG")
                pngs.append(buffer.getvalue())
        return pngs
    finally:
        pdf.close()


def clean(text: str) -> str:
    text = _SPECIAL.sub("", _GROUNDING.sub("", text))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def main() -> None:
    cli = parser(__doc__)
    cli.add_argument("--url", default="http://127.0.0.1:18000", help="SGLang server")
    cli.add_argument("--model", default="baidu/Unlimited-OCR-2", help="Served model name")
    cli.add_argument("--tool", default="unlimited-ocr-2-sglang", help="Name of the results folder")
    cli.add_argument("--repeats", type=int, default=1, help="Conversions per document; the median time is kept")
    cli.add_argument(
        "--logit-processor",
        type=Path,
        default=Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "unlimited-ocr-2" / "ngram_processor.txt",
        help="The model card's no-repeat n-gram processor, serialised by the server's setup",
    )  # fmt: skip
    args = cli.parse_args()
    processor = args.logit_processor.read_text(encoding="utf-8")
    recorder = Recorder(args.tool, args.results, args.force)
    client = httpx.Client(timeout=httpx.Timeout(3600, connect=30))
    recorder.setup(model_load_s=0.0, endpoint=args.url, model=args.model)

    def convert_once(pngs: list[bytes]) -> dict:
        images = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}
            for png in pngs
        ]
        prompt = "<PAGE><image>" * len(pngs) + "Multi page parsing."
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": [*images, {"type": "text", "text": prompt}]}],
            "max_tokens": TOKENS_PER_PAGE * len(pngs),
            "temperature": 0,
            "skip_special_tokens": False,
            "images_config": {"crop_mode": len(pngs) == 1},
            "custom_logit_processor": processor,
            "custom_params": {
                "ngram_size": 35,
                "window_size": 1024 if len(pngs) > 1 else 128,
                "page_token_id": PAGE_TOKEN_ID,
                "eos_token_id": 1,
                "max_page_tokens": 20000,
            },
        }
        t_start = time.perf_counter()
        response = client.post(f"{args.url}/v1/chat/completions", json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
        body = response.json()
        raw = body["choices"][0]["message"]["content"] or ""
        # Text before the first <PAGE> belongs to page 1
        pages = [clean(part) for part in raw.split("<PAGE>")]
        if len(pages) > len(pngs) and not pages[0]:
            pages = pages[1:]
        return {
            "markdown": "\n\n".join(pages),
            "pages": pages,
            "seconds": time.perf_counter() - t_start,
            "meta": {
                "page_images": len(pngs),
                "pages_in_answer": len(pages),
                "finish_reason": body["choices"][0].get("finish_reason"),
                "usage": body.get("usage"),
                "raw": raw[:20000],
            },
        }

    def convert(doc: Path) -> dict:
        pngs = page_images(doc)
        runs = [convert_once(pngs) for _ in range(args.repeats)]
        out = runs[0]
        out["meta"]["wall_runs_s"] = [r["seconds"] for r in runs]
        out["seconds"] = statistics.median(r["seconds"] for r in runs)
        return out

    recorder.run(list_docs(args.docs, args.only), convert)


if __name__ == "__main__":
    main()
