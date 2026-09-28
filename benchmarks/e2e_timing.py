# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx", "pypdfium2"]
# ///
"""End-to-end processing time of docling-serve images vs. plain docling.

Every variant runs on its own (one container at a time), gets one untimed-for-the-summary
cold conversion to load its models, then converts every document --repeats times.
For the docling-serve variants the wall time spans upload -> result download through the
async API, which is what API clients experience.

    uv run --script benchmarks/e2e_timing.py --docs ../test-docs
    uv run --script benchmarks/e2e_timing.py --docs ../test-docs --variants official-cu130 custom-cu130
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
import pypdfium2

HERE = Path(__file__).resolve().parent

OPTIONS = {
    "do_ocr": True,
    "pdf_backend": "docling_parse",
    "table_mode": "accurate",
    "ocr_preset": "rapidocr",
    "ocr_lang": ["de"],
}

# name -> (image without tag, needs GPU); None marks the non-Docker docling run.
VARIANTS: dict[str, tuple[str | None, bool]] = {
    "official-cu130": ("ghcr.io/docling-project/docling-serve-cu130", True),
    "custom-cu130": ("ghcr.io/dcc-bs/dcc-docling-serve-cu130", True),
    "local-docling": (None, True),
    "official-cpu": ("ghcr.io/docling-project/docling-serve-cpu", False),
    "custom-cpu": ("ghcr.io/dcc-bs/dcc-docling-serve-cpu", False),
}

ENVIRONMENT_PROBE = """
import importlib.metadata, json
info = {}
for p in ("docling-serve", "docling", "docling-slim", "docling-core", "rapidocr", "onnxruntime", "onnxruntime-gpu", "torch"):
    try:
        info[p] = importlib.metadata.version(p)
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
print(json.dumps(info))
"""

SUPPORTED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", file=sys.stderr, flush=True)


def page_count(path: Path) -> int:
    if path.suffix.lower() != ".pdf":
        return 1
    pdf = pypdfium2.PdfDocument(path)
    try:
        return len(pdf)
    finally:
        pdf.close()


def normalize_timings(timings: dict | None) -> dict:
    """Server timings carry every sample; keep count and total per stage like the local runner."""
    return {
        name: {"scope": item.get("scope"), "count": item.get("count"), "total": sum(item.get("times") or [])}
        for name, item in (timings or {}).items()
    }


class ServeRunner:
    def __init__(self, variant: str, image: str, gpu: bool, args: argparse.Namespace):
        self.variant = variant
        self.image = image
        self.gpu = gpu
        self.args = args
        self.container = f"docling-bench-{variant}-{os.getpid()}"
        self.base_url = f"http://127.0.0.1:{args.port}"
        self.client = httpx.Client(base_url=self.base_url, timeout=httpx.Timeout(600, connect=10))

    def start(self) -> float:
        if self.args.pull:
            log(f"{self.variant}: pulling {self.image}")
            subprocess.run(["docker", "pull", "-q", self.image], check=True, stdout=subprocess.DEVNULL)
        command = ["docker", "run", "-d", "--rm", "--name", self.container, "-p", f"127.0.0.1:{self.args.port}:5001"]
        if self.gpu:
            command += ["--gpus", f"device={self.args.gpu_device}"]
        command += ["-e", "DOCLING_DEBUG_PROFILE_PIPELINE_TIMINGS=true", self.image]
        t0 = time.perf_counter()
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        deadline = t0 + self.args.startup_timeout
        while time.perf_counter() < deadline:
            try:
                if self.client.get("/health", timeout=5).status_code == 200:
                    return time.perf_counter() - t0
            except httpx.HTTPError:
                pass
            time.sleep(1)
        raise TimeoutError(f"{self.variant} did not become healthy within {self.args.startup_timeout}s")

    def environment(self) -> dict:
        result = subprocess.run(
            ["docker", "exec", self.container, "python", "-c", ENVIRONMENT_PROBE],
            capture_output=True,
            text=True,
        )
        lines = result.stdout.strip().splitlines()
        return json.loads(lines[-1]) if lines else {"error": result.stderr[-500:]}

    def convert(self, path: Path) -> dict:
        data = {
            key: [str(item) for item in value] if isinstance(value, list) else json.dumps(value).strip('"')
            for key, value in OPTIONS.items()
        }
        t0 = time.perf_counter()
        with path.open("rb") as handle:
            response = self.client.post("/v1/convert/file/async", files={"files": (path.name, handle)}, data=data)
        response.raise_for_status()
        task_id = response.json()["task_id"]

        deadline = t0 + self.args.request_timeout
        status = None
        while time.perf_counter() < deadline:
            status = self.client.get(f"/v1/status/poll/{task_id}").json().get("task_status")
            if status in ("success", "failure"):
                break
            time.sleep(self.args.poll_interval)
        else:
            return {"status": "timeout", "wall_s": time.perf_counter() - t0}

        result = self.client.get(f"/v1/result/{task_id}")
        wall = time.perf_counter() - t0
        if result.status_code != 200:
            return {"status": f"http_{result.status_code}", "wall_s": wall, "errors": result.text[-500:]}
        body = result.json()
        return {
            "status": body.get("status", status),
            "wall_s": wall,
            "server_processing_s": body.get("processing_time"),
            "timings": normalize_timings(body.get("timings")),
            "errors": body.get("errors") or None,
        }

    def run(self, docs: list[Path], warmup: Path, record) -> None:
        try:
            startup = self.start()
            log(f"{self.variant}: healthy after {startup:.1f}s")
            record({"kind": "startup", "startup_s": startup})
            record({"kind": "environment", **self.environment()})
            log(f"{self.variant}: cold conversion of {warmup.name}")
            record({"kind": "cold", "doc": warmup.name, **self.convert(warmup)})
            for repeat in range(self.args.repeats):
                for doc in docs:
                    outcome = self.convert(doc)
                    log(f"{self.variant}: {doc.name} #{repeat + 1} {outcome['status']} in {outcome['wall_s']:.1f}s")
                    record({"kind": "run", "doc": doc.name, "repeat": repeat, **outcome})
        finally:
            self.client.close()
            subprocess.run(["docker", "rm", "-f", self.container], capture_output=True)


def run_local(docs: list[Path], warmup: Path, args: argparse.Namespace, record) -> None:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu_device)}
    command = [
        "uv", "run", "--script", "--upgrade", str(HERE / "local_docling.py"),
        "--options", json.dumps(OPTIONS), "--repeats", str(args.repeats), "--warmup", str(warmup),
        *map(str, docs),
    ]  # fmt: skip
    log("local-docling: installing latest docling and converting (first run downloads models)")
    t0 = time.perf_counter()
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, text=True)
    first = True
    for line in process.stdout:
        line = line.strip()
        if not line.startswith("{"):
            continue
        event = json.loads(line)
        if first:
            # Interpreter start plus imports, comparable to a container becoming healthy.
            record({"kind": "startup", "startup_s": time.perf_counter() - t0})
            first = False
        if event["kind"] == "run":
            log(f"local-docling: {event['doc']} #{event['repeat'] + 1} {event['status']} in {event['wall_s']:.1f}s")
        record(event)
    if process.wait() != 0:
        raise RuntimeError(f"local docling runner exited with {process.returncode}")


def summarize(records: list[dict], pages: dict[str, int], variants: list[str]) -> str:
    by_variant = {v: [r for r in records if r["variant"] == v] for v in variants}
    ok_statuses = {"success", "partial_success"}

    def runs(variant: str) -> list[dict]:
        return [r for r in by_variant[variant] if r["kind"] == "run"]

    def median_wall(variant: str, doc: str) -> float | None:
        walls = [r["wall_s"] for r in runs(variant) if r["doc"] == doc and r["status"] in ok_statuses]
        return statistics.median(walls) if walls else None

    def first(variant: str, kind: str) -> dict:
        return next((r for r in by_variant[variant] if r["kind"] == kind), {})

    docs = list(pages)
    total_pages = sum(pages.values())
    lines = ["# Docling end-to-end timing", "", f"Options: `{json.dumps(OPTIONS)}`", ""]
    lines += [f"Documents: {len(docs)} ({total_pages} pages)", ""]

    lines += ["## Overview", "", "Warm total = sum over documents of the median wall time per document.", ""]
    lines += [
        "| Variant | Startup s | Cold first doc s | Warm total s | s / page | Server processing s | Failed runs |"
    ]
    lines += ["|---|---:|---:|---:|---:|---:|---:|"]
    for v in variants:
        medians = [median_wall(v, d) for d in docs]
        complete = all(m is not None for m in medians)
        warm_total = sum(m for m in medians if m is not None)
        processing = [r.get("server_processing_s") for r in runs(v) if r.get("server_processing_s") is not None]
        failed = sum(1 for r in runs(v) if r["status"] not in ok_statuses)
        cold = first(v, "cold").get("wall_s")
        lines.append(
            f"| {v} | {first(v, 'startup').get('startup_s', float('nan')):.1f} "
            f"| {cold if cold is not None else float('nan'):.1f} "
            f"| {warm_total:.1f}{'' if complete else ' (incomplete)'} "
            f"| {warm_total / total_pages if complete and total_pages else float('nan'):.2f} "
            f"| {statistics.mean(processing) if processing else float('nan'):.1f} (mean) "
            f"| {failed} |"
        )

    lines += ["", "## Median wall time per document (s)", ""]
    lines += ["| Document | Pages | " + " | ".join(variants) + " |", "|---|---:|" + "---:|" * len(variants)]
    for d in docs:
        cells = [median_wall(v, d) for v in variants]
        lines.append(f"| {d} | {pages[d]} | " + " | ".join("-" if c is None else f"{c:.1f}" for c in cells) + " |")

    stage_totals: dict[str, dict[str, float]] = {}
    for v in variants:
        variant_runs = [r for r in runs(v) if r.get("timings")]
        repeats = max(1, len({r["repeat"] for r in variant_runs}))
        for r in variant_runs:
            for stage, item in r["timings"].items():
                stage_totals.setdefault(stage, {}).setdefault(v, 0.0)
                stage_totals[stage][v] += (item.get("total") or 0.0) / repeats
    if stage_totals:
        lines += ["", "## Pipeline stage time per pass over all documents (s)", ""]
        lines += [
            "Summed stage timings reported by docling, averaged over repeats. Stages run concurrently, so "
            "they do not add up to the wall time.",
            "",
        ]
        lines += ["| Stage | " + " | ".join(variants) + " |", "|---|" + "---:|" * len(variants)]
        ordered = sorted(stage_totals, key=lambda s: -max(stage_totals[s].values()))
        for stage in ordered:
            cells = [stage_totals[stage].get(v) for v in variants]
            lines.append(f"| {stage} | " + " | ".join("-" if c is None else f"{c:.1f}" for c in cells) + " |")

    lines += ["", "## Environment", ""]
    for v in variants:
        env = {k: val for k, val in first(v, "environment").items() if k not in ("kind", "variant")}
        lines.append(f"- **{v}**: `{json.dumps(env)}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", type=Path, required=True, help="Folder with the documents to convert")
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--tag", default="v1.35.0", help="Tag of all docling-serve images (default: %(default)s)")
    parser.add_argument("--repeats", type=int, default=3, help="Timed conversions per document (default: %(default)s)")
    parser.add_argument("--gpu-device", default="0", help="GPU index for the GPU variants (default: %(default)s)")
    parser.add_argument("--port", type=int, default=5091, help="Host port for the containers (default: %(default)s)")
    parser.add_argument("--poll-interval", type=float, default=0.25, help="Task status poll interval in s")
    parser.add_argument("--startup-timeout", type=float, default=900, help="Seconds to wait for /health")
    parser.add_argument("--request-timeout", type=float, default=3600, help="Seconds per conversion")
    parser.add_argument("--pull", action="store_true", help="docker pull each image before starting it")
    parser.add_argument("--out", type=Path, default=HERE / "results", help="Results folder (default: %(default)s)")
    args = parser.parse_args()

    docs = sorted(p.resolve() for p in args.docs.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)
    if not docs:
        sys.exit(f"No supported documents in {args.docs}")
    pages = {p.name: page_count(p) for p in docs}
    # Warm up on the smallest document so model loading is not mixed into the timed runs.
    warmup = min(docs, key=lambda p: p.stat().st_size)

    out_dir = args.out / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True)
    raw_path = out_dir / "raw.jsonl"
    records: list[dict] = []

    for variant in args.variants:

        def record(event: dict, variant: str = variant) -> None:
            event = {"variant": variant, **event}
            records.append(event)
            with raw_path.open("a") as handle:
                handle.write(json.dumps(event) + "\n")

        image, gpu = VARIANTS[variant]
        try:
            if image is None:
                run_local(docs, warmup, args, record)
            else:
                ServeRunner(variant, f"{image}:{args.tag}", gpu, args).run(docs, warmup, record)
        except Exception as error:  # keep going so one broken variant does not lose the others
            log(f"{variant}: FAILED: {error}")
            record({"kind": "error", "error": str(error)})

    summary = summarize(records, pages, args.variants)
    (out_dir / "summary.md").write_text(summary)
    print(summary)
    log(f"results in {out_dir}")


if __name__ == "__main__":
    main()
