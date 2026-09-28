# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx"]
# ///
"""Convert the documents with a docling-serve image (one container, async API).

    uv run --script run_docling.py --tool docling-today --image ghcr.io/docling-project/docling-serve-cu130:v1.35.0
    uv run --script run_docling.py --tool docling-gpu-ocr --image dcc-docling-serve-test:v1.35.0-cu130 \
        --env DCC_OCR_IGNORE_SHAPES=0
"""

import json
import os
import subprocess
import time

import httpx
from common import Recorder, list_docs, log, parser

PAGE_BREAK = "<!-- dcc-page-break -->"
OPTIONS = {
    "do_ocr": "true",
    "pdf_backend": "docling_parse",
    "table_mode": "accurate",
    "ocr_preset": "rapidocr",
    "ocr_lang": ["de"],
    "to_formats": ["md"],
    "image_export_mode": "placeholder",
    "md_page_break_placeholder": PAGE_BREAK,
}


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--tool", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--env", action="append", default=[], help="KEY=VALUE passed to the container")
    p.add_argument("--gpu-device", default="0")
    p.add_argument("--port", type=int, default=5093)
    p.add_argument("--timeout", type=float, default=7200)
    args = p.parse_args()

    recorder = Recorder(args.tool, args.results, args.force)
    docs = list_docs(args.docs, args.only)
    container = f"tool-comparison-{args.tool}-{os.getpid()}"
    command = ["docker", "run", "-d", "--rm", "--name", container, "-p", f"127.0.0.1:{args.port}:5001",
               "--gpus", f"device={args.gpu_device}"]  # fmt: skip
    for item in args.env:
        command += ["-e", item]
    client = httpx.Client(base_url=f"http://127.0.0.1:{args.port}", timeout=httpx.Timeout(900, connect=10))
    t0 = time.perf_counter()
    subprocess.run([*command, args.image], check=True, stdout=subprocess.DEVNULL)
    try:
        while True:
            try:
                if client.get("/health", timeout=5).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(1)
        startup = time.perf_counter() - t0

        def convert(doc):
            t_start = time.perf_counter()
            with doc.open("rb") as handle:
                response = client.post("/v1/convert/file/async", files={"files": (doc.name, handle)}, data=OPTIONS)
            response.raise_for_status()
            task_id = response.json()["task_id"]
            while True:
                status = client.get(f"/v1/status/poll/{task_id}").json().get("task_status")
                if status in ("success", "failure"):
                    break
                if time.perf_counter() - t_start > args.timeout:
                    raise TimeoutError(f"no result after {args.timeout}s")
                time.sleep(0.25)
            result = client.get(f"/v1/result/{task_id}")
            result.raise_for_status()
            body = result.json()
            if body.get("status") not in ("success", "partial_success"):
                raise RuntimeError(f"docling status {body.get('status')}: {body.get('errors')}")
            markdown = body["document"]["md_content"] or ""
            return {
                "markdown": markdown.replace(PAGE_BREAK, "\n\n"),
                "pages": markdown.split(PAGE_BREAK),
                "meta": {"server_processing_s": body.get("processing_time"), "docling_status": body.get("status")},
            }

        # Load the models before timing anything.
        warm_t0 = time.perf_counter()
        convert(docs[0])
        recorder.setup(startup_s=startup, first_conversion_s=time.perf_counter() - warm_t0, image=args.image,
                       env=args.env)  # fmt: skip
        log(f"{args.tool}: healthy after {startup:.1f}s")
        recorder.run(docs, convert)
    finally:
        client.close()
        subprocess.run(["docker", "rm", "-f", container], capture_output=True)
    print(json.dumps({"tool": args.tool, "done": len(docs)}))


if __name__ == "__main__":
    main()
