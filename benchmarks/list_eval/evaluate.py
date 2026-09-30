# /// script
# requires-python = ">=3.12"
# dependencies = ["pypdfium2"]
# ///
"""Score the list benchmark (run_all.sh).

Ground-truth documents (docs/*.gt.json): recall and precision of list items, and traps
(headings, dates, amounts, footnotes) that came out as list items. Real documents
(TEST_DOCS): recall of text-layer lines that start with a bullet glyph, and agreement with
Heron's list items (Heron is not ground truth; on legal texts it also lists numbered
paragraphs and section numbers).

    uv run --script benchmarks/list_eval/evaluate.py <results dir>
"""

import json
import os
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import pypdfium2 as pdfium

HERE = Path(__file__).parent
TEST_DOCS = Path(os.environ.get("TEST_DOCS", HERE / "../../../test-docs"))
MARK = re.compile(r"^\s*(?:[•·▪▫◦○●■□►▸‣⁃✓✔→➢\-–—*]\s*|\(?(?:\d{1,2}|[a-zA-Z]|[ivxIVX]{2,5})[.)]\s*)")
BULLET_LINE = re.compile(r"^\s*[•·▪◦●■‣⁃–]\s*\S")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", MARK.sub("", s or "", count=1)).strip().lower()


def same(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b or (min(len(a), len(b)) >= 12 and (a.startswith(b[:40]) or b.startswith(a[:40]))):
        return True
    return SequenceMatcher(None, a[:120], b[:120]).ratio() >= 0.85


class Index:
    """Texts bucketed by their first 6 characters; fuzzy matching only within a bucket."""

    def __init__(self, texts):
        self.buckets = {}
        for t in texts:
            self.buckets.setdefault(t[:6], []).append(t)

    def has(self, text):
        return any(same(text, t) for t in self.buckets.get(text[:6], []))


def matches(needles, hay):
    idx = Index(hay)
    return sum(1 for n in needles if idx.has(n))


def list_items(record):
    return [norm(text) for label, text, *_ in record["items"] if label == "list_item"]


def bullet_lines(doc):
    path = TEST_DOCS / doc
    if path.suffix.lower() != ".pdf" or not path.exists():
        return []
    pdf = pdfium.PdfDocument(path)
    lines = (ln for i in range(len(pdf)) for ln in pdf[i].get_textpage().get_text_range().splitlines())
    return [norm(ln) for ln in lines if BULLET_LINE.match(ln) and len(norm(ln)) >= 8]


def main():
    results = Path(sys.argv[1])
    configs = sorted(p.name for p in results.iterdir() if p.is_dir())
    load = {c: {p.name[:-5]: json.loads(p.read_text()) for p in (results / c).glob("*.json")} for c in configs}
    gt_docs = sorted(p.name.replace(".gt.json", ".pdf") for p in (HERE / "docs").glob("*.gt.json"))
    real = sorted(d for d in load.get("heron", load[configs[0]]) if d not in gt_docs)
    refs = {d: bullet_lines(d) for d in real}

    print(f"{'config':10} | {'digital R/P':>11} | {'scan R/P':>11} | traps | {'bullets':>9} | "
          f"{'vs Heron R/P':>12} | {'items':>5} | time")  # fmt: skip
    for c in configs:
        res = load[c]
        scores = {}
        traps = 0
        for kind in ("digital", "scan"):
            g = f = n = ok = 0
            for d in gt_docs:
                if ("scan" in d) != (kind == "scan") or d not in res:
                    continue
                gt = json.loads((HERE / "docs" / d.replace(".pdf", ".gt.json")).read_text())
                items = [norm(i["text"]) for i in gt["items"]]
                li = list_items(res[d])
                gi = Index(items)
                g, f, n = g + len(items), f + matches(items, li), n + len(li)
                ok += sum(1 for x in li if gi.has(x))
                traps += matches(li, [norm(t) for t in gt["traps"]])
            scores[kind] = f"{f / max(g, 1):.2f}/{ok / max(n, 1):.2f}"
        b = rf = rh = hn = ph = n = 0
        seconds = 0.0
        for d in real:
            if d not in res:
                continue
            li, hl = list_items(res[d]), list_items(load["heron"][d]) if "heron" in load else []
            b, rf = b + matches(refs[d], li), rf + len(refs[d])
            rh, hn, ph, n = rh + matches(hl, li), hn + len(hl), ph + matches(li, hl), n + len(li)
            seconds += res[d]["seconds"]
        print(f"{c:10} | {scores['digital']:>11} | {scores['scan']:>11} | {traps:5} | {b:4}/{rf:<4} | "
              f"{rh / max(hn, 1):.2f}/{ph / max(n, 1):.2f}{'':3} | {n:5} | {seconds:.0f}s")  # fmt: skip


if __name__ == "__main__":
    main()
