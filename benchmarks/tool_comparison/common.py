"""Shared helpers for the tool runners (standard library only, imported by every runner).

Every runner writes one JSON file per document to <results>/outputs/<tool>/<document>.json:
    tool, doc, status (ok | error | unsupported | timeout), seconds (warm conversion time),
    markdown, pages (list of per-page markdown, or null when the tool has no page split),
    error, meta
Model loading is timed once per tool and written to <tool>/_setup.json.
"""

import argparse
import json
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUPPORTED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
DEFAULT_DOCS = [HERE.parents[2] / "test-docs", HERE / "synthetic"]
DEFAULT_RESULTS = HERE / "results"


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--docs", type=Path, nargs="+", default=DEFAULT_DOCS, help="Folders with documents")
    p.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--only", nargs="*", help="Only these document file names")
    p.add_argument("--force", action="store_true", help="Redo documents that already have a result")
    return p


def list_docs(folders: list[Path], only: list[str] | None = None) -> list[Path]:
    """Documents in the given folders (single files are accepted too), smallest first."""
    docs = sorted(
        (
            p
            for folder in folders
            for p in (folder.iterdir() if folder.is_dir() else [folder])
            if p.suffix.lower() in SUPPORTED_SUFFIXES
        ),
        key=lambda p: p.stat().st_size,
    )
    return [d for d in docs if not only or d.name in only]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


class Recorder:
    def __init__(self, tool: str, results: Path, force: bool = False):
        self.tool, self.force = tool, force
        self.dir = results / "outputs" / tool
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, doc: Path) -> Path:
        return self.dir / f"{doc.name}.json"

    def done(self, doc: Path) -> bool:
        path = self.path(doc)
        return not self.force and path.exists() and json.loads(path.read_text()).get("status") == "ok"

    def write(self, doc: Path, **fields) -> None:
        record = {"tool": self.tool, "doc": doc.name, "status": "ok", "seconds": None, "markdown": "",
                  "pages": None, "error": None, "meta": {}} | fields  # fmt: skip
        self.path(doc).write_text(json.dumps(record, ensure_ascii=False))

    def setup(self, **fields) -> None:
        (self.dir / "_setup.json").write_text(json.dumps({"tool": self.tool, **fields}, ensure_ascii=False))

    def run(self, docs: list[Path], convert: Callable[[Path], dict]) -> None:
        """convert(doc) returns {"markdown", "pages", "meta"} and may raise."""
        for doc in docs:
            if self.done(doc):
                continue
            t0 = time.perf_counter()
            try:
                out = convert(doc)
                seconds = time.perf_counter() - t0
                self.write(doc, seconds=seconds, **out)
                log(f"{self.tool}: {doc.name} ok in {seconds:.1f}s")
            except Exception as error:  # keep going: one bad document must not stop the run
                seconds = time.perf_counter() - t0
                self.write(doc, status="error", seconds=seconds, error=f"{type(error).__name__}: {error}",
                           meta={"traceback": traceback.format_exc()[-2000:]})  # fmt: skip
                log(f"{self.tool}: {doc.name} FAILED after {seconds:.1f}s: {error}")
