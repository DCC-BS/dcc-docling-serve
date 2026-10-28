# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx"]
# ///
"""Convert the documents with a docling-serve image (one container, async API).

uv run --script run_docling.py --tool docling-today --image ghcr.io/docling-project/docling-serve-cu130:v1.35.0
uv run --script run_docling.py --tool docling-gpu-ocr --image dcc-docling-serve-test:v1.38.1-cu130
"""

import json
import os
import statistics
import subprocess
import time
from pathlib import Path

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


def stage_totals(timings: dict | None) -> dict:
    """docling-serve reports every sample per stage; keep the count and the total."""
    return {
        name: {"count": item.get("count"), "total_s": sum(item.get("times") or [])}
        for name, item in (timings or {}).items()
        if isinstance(item, dict)
    }


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--tool", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--env", action="append", default=[], help="KEY=VALUE passed to the container")
    p.add_argument("--option", action="append", default=[], metavar="KEY=VALUE",
                   help="Extra or overriding conversion option, e.g. layout_preset=ppdoclayout-v3")  # fmt: skip
    p.add_argument("--docker-arg", action="append", default=[], help="Extra argument for docker run")
    p.add_argument("--gpu-device", default="0")
    p.add_argument("--port", type=int, default=5093)
    p.add_argument("--timeout", type=float, default=7200)
    p.add_argument("--repeats", type=int, default=1, help="Convert every document this often and keep the median time")
    p.add_argument("--warmup", type=Path, nargs="*", default=None,
                   help="Untimed conversions before the run (default: the first document)")  # fmt: skip
    args = p.parse_args()

    recorder = Recorder(args.tool, args.results, args.force)
    docs = list_docs(args.docs, args.only)
    container = f"tool-comparison-{args.tool}-{os.getpid()}"
    command = ["docker", "run", "-d", "--rm", "--name", container, "-p", f"127.0.0.1:{args.port}:5001",
               "--gpus", f"device={args.gpu_device}", "-e", "DOCLING_DEBUG_PROFILE_PIPELINE_TIMINGS=true"]  # fmt: skip
    for item in args.env:
        command += ["-e", item]
    command += args.docker_arg
    options = dict(OPTIONS)
    for item in args.option:
        key, _, value = item.partition("=")
        options[key] = value
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

        def convert_once(doc):
            t_start = time.perf_counter()
            with doc.open("rb") as handle:
                response = client.post("/v1/convert/file/async", files={"files": (doc.name, handle)}, data=options)
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
            if body.get("status") == "partial_success":
                # Kept and scored, but flagged: docling dropped part of the document (pages, a VLM answer)
                log(f"{doc.name}: partial_success: {body.get('errors')}")
            return {
                "markdown": markdown.replace(PAGE_BREAK, "\n\n"),
                "pages": markdown.split(PAGE_BREAK),
                "seconds": time.perf_counter() - t_start,
                "meta": {
                    "server_processing_s": body.get("processing_time"),
                    "docling_status": body.get("status"),
                    "docling_errors": body.get("errors") or None,
                    "timings": stage_totals(body.get("timings")),
                },
            }

        def convert(doc):
            runs = [convert_once(doc) for _ in range(args.repeats)]
            out = runs[0]
            # A partial result in any repeat flags the document: the median includes that run
            partial = [r["meta"] for r in runs if r["meta"]["docling_status"] == "partial_success"]
            if partial:
                out["meta"] |= {"docling_status": "partial_success", "docling_errors": partial[0]["docling_errors"]}
            server = [r["meta"]["server_processing_s"] for r in runs if r["meta"]["server_processing_s"] is not None]
            out["meta"] |= {
                "server_processing_s": statistics.median(server) if server else None,
                "server_runs_s": server,
                "wall_runs_s": [r["seconds"] for r in runs],
                "load1": os.getloadavg()[0],
            }
            out["seconds"] = statistics.median(r["seconds"] for r in runs)
            return out

        # Load the models before timing anything; with --warmup, documents that reach every stage
        warm_t0 = time.perf_counter()
        for warm_doc in args.warmup or docs[:1]:
            convert_once(warm_doc)
        recorder.setup(startup_s=startup, first_conversion_s=time.perf_counter() - warm_t0, image=args.image,
                       env=args.env, options=args.option, repeats=args.repeats,
                       warmup=[str(d) for d in args.warmup or docs[:1]])  # fmt: skip
        log(f"{args.tool}: healthy after {startup:.1f}s")
        recorder.run(docs, convert)
    finally:
        client.close()
        subprocess.run(["docker", "rm", "-f", container], capture_output=True)
    print(json.dumps({"tool": args.tool, "done": len(docs)}))


if __name__ == "__main__":
    main()
