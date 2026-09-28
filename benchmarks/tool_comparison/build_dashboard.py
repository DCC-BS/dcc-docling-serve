# /// script
# requires-python = ">=3.12"
# dependencies = ["pypdfium2", "pillow"]
# ///
"""Score every tool's markdown and build the comparison dashboard.

Scores:
  - synthetic documents: against the exact ground truth written by make_synthetic.py
    (body text F1, headings, tables, reading order, and recall per text source:
    chart labels, outlined text, bitmap text, scanned text)
  - real PDFs: against the PDF's own text layer (word recall / precision). Pages without a
    text layer (scans, photos) have no reference and are left out of that score.

Output: <results>/dashboard/index.html plus data/ and pages/ next to it.

    uv run --script build_dashboard.py
"""

import argparse
import difflib
import html
import json
import re
import shutil
import statistics
import unicodedata
from collections import Counter
from pathlib import Path

import pypdfium2
from common import DEFAULT_DOCS, DEFAULT_RESULTS, HERE, list_docs
from PIL import Image

TOOLS = {
    "docling-today": "docling (today)",
    "docling-gpu-ocr": "docling + GPU OCR",
    "docling-gpu-ocr-noshape": "docling + GPU OCR + shape patch",
    "marker": "marker",
    "doctr": "docTR",
    "markitdown": "markitdown",
    "jina": "Jina Reader",
}
SOURCE_LABELS = {
    "body": "Body text",
    "heading": "Headings (text)",
    "table": "Table text",
    "chart_text": "Chart labels (real text)",
    "outlined_text": "Outlined text (vector paths)",
    "bitmap_text": "Text in embedded images",
    "scan_text": "Scanned page text",
    "furniture": "Page header/footer",
}
PAGE_WIDTH_PX = 1100


# ------------------------------------------------------------------ text normalisation
def to_plain(markdown: str) -> str:
    text = re.sub(r"<!--.*?-->", " ", markdown, flags=re.S)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)  # images
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links keep their text
    # html tags only: a bare "<" in maths ("p < 1") must not swallow text up to the next ">"
    text = re.sub(r"</?[A-Za-z][A-Za-z0-9]*(?:\s[^<>\n]{0,300})?/?>", " ", text)
    text = re.sub(r"\{\d+\}-{10,}", " ", text)  # marker page separators
    text = re.sub(r"(?m)^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$", " ", text)  # table rules
    return html.unescape(text)


def tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    # Re-join words hyphenated at line ends in PDF text layers ("Verhält-\nnis")
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    return re.findall(r"\w+", text)


def bag_scores(output: list[str], reference: list[str]) -> dict:
    out, ref = Counter(output), Counter(reference)
    overlap = sum((out & ref).values())
    recall = overlap / sum(ref.values()) if ref else None
    precision = overlap / sum(out.values()) if out else (None if not ref else 0.0)
    f1 = 2 * recall * precision / (recall + precision) if recall and precision else 0.0 if ref else None
    return {"recall": recall, "precision": precision, "f1": f1}


def snippet_recall(snippet: str, out: Counter) -> float:
    need = Counter(tokens(snippet))
    return sum((need & out).values()) / sum(need.values()) if need else 1.0


# ------------------------------------------------------------------ structure parsing
def markdown_tables(markdown: str) -> list[list[list[str]]]:
    tables, current = [], []
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.count("|") >= 2:
            if re.fullmatch(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?", stripped):
                continue
            current.append([" ".join(tokens(to_plain(c))) for c in stripped.strip("|").split("|")])
        elif current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    for body in re.findall(r"<table.*?</table>", markdown, flags=re.S | re.I):  # html tables (marker, docTR)
        rows = [
            [" ".join(tokens(to_plain(c))) for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, flags=re.S | re.I)]
            for row in re.findall(r"<tr.*?</tr>", body, flags=re.S | re.I)
        ]
        tables.append([r for r in rows if r])
    return [t for t in tables if t]


def table_score(gt_tables, out_tables) -> float | None:
    """Mean over ground-truth rows of the best cell overlap with any output table row."""
    if not gt_tables:
        return None
    out_rows = [set(filter(None, row)) for table in out_tables for row in table]
    scores = []
    for table in gt_tables:
        for row in table:
            cells = set(filter(None, row))
            if not cells:
                continue
            best = max((len(cells & r) / len(cells) for r in out_rows), default=0.0)
            scores.append(best)
    return statistics.mean(scores) if scores else None


def headings(markdown: str) -> list[str]:
    return [" ".join(tokens(to_plain(m))) for m in re.findall(r"(?m)^\s{0,3}#{1,6}\s+(.+)$", markdown)]


def heading_score(gt: list[str], out: list[str]) -> float | None:
    if not gt:
        return None
    hits = sum(1 for h in gt if any(difflib.SequenceMatcher(None, h, o).ratio() >= 0.85 for o in out))
    return hits / len(gt)


def order_score(gt_tokens: list[str], out_tokens: list[str]) -> float | None:
    """Similarity of token order, restricted to words that exist in the ground truth."""
    vocab = set(gt_tokens)
    seq = [t for t in out_tokens if t in vocab]
    if not gt_tokens:
        return None
    return difflib.SequenceMatcher(None, gt_tokens, seq, autojunk=False).ratio()


def structure(markdown: str) -> dict:
    return {
        "headings": len(headings(markdown)),
        "tables": len(markdown_tables(markdown)),
        "images": len(re.findall(r"<!--\s*image\s*-->|!\[[^\]]*\]\(", markdown)),
        "chars": len(markdown),
    }


# ------------------------------------------------------------------ documents
def doc_id(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def load_outputs(results: Path) -> dict[str, dict[str, dict]]:
    outputs: dict[str, dict[str, dict]] = {}
    for tool in TOOLS:
        folder = results / "outputs" / tool
        if not folder.exists():
            continue
        for path in folder.glob("*.json"):
            if path.name == "_setup.json":
                continue
            record = json.loads(path.read_text())
            outputs.setdefault(record["doc"], {})[tool] = record
    return outputs


def load_setups(results: Path) -> dict[str, dict]:
    setups = {}
    for tool in TOOLS:
        path = results / "outputs" / tool / "_setup.json"
        if path.exists():
            setups[tool] = json.loads(path.read_text())
    return setups


def render_pages(doc: Path, target: Path) -> tuple[list[str], list[str]]:
    """Render page images; return (image paths relative to dashboard, text layer per page)."""
    target.mkdir(parents=True, exist_ok=True)
    images, texts = [], []
    if doc.suffix.lower() != ".pdf":
        img = Image.open(doc)
        img.thumbnail((PAGE_WIDTH_PX, PAGE_WIDTH_PX * 2))
        out = target / "1.jpg"
        if not out.exists():
            img.convert("RGB").save(out, quality=70)
        return [f"pages/{target.name}/1.jpg"], [""]
    pdf = pypdfium2.PdfDocument(doc)
    for index in range(len(pdf)):
        page = pdf[index]
        texts.append(page.get_textpage().get_text_range())
        out = target / f"{index + 1}.jpg"
        if not out.exists():
            width = page.get_width()
            page.render(scale=min(2.0, PAGE_WIDTH_PX / width)).to_pil().convert("RGB").save(out, quality=70)
        images.append(f"pages/{target.name}/{index + 1}.jpg")
    pdf.close()
    return images, texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", type=Path, nargs="+", default=DEFAULT_DOCS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    out_dir = args.results / "dashboard"
    (out_dir / "data" / "docs").mkdir(parents=True, exist_ok=True)
    outputs, setups = load_outputs(args.results), load_setups(args.results)
    tools = [t for t in TOOLS if t in setups or any(t in o for o in outputs.values())]
    synthetic_dir = HERE / "synthetic"

    doc_rows = []
    for doc in list_docs(args.docs):
        print("scoring", doc.name, flush=True)
        did = doc_id(doc.name)
        images, layer_texts = render_pages(doc, out_dir / "pages" / did)
        gt_path = synthetic_dir / f"{doc.stem}.gt.md"
        synthetic = gt_path.exists() and doc.parent.resolve() == synthetic_dir.resolve()
        ground_truth = gt_path.read_text() if synthetic else None
        checks = json.loads((synthetic_dir / f"{doc.stem}.checks.json").read_text()) if synthetic else []
        layer_tokens = [tokens(t) for t in layer_texts]
        text_pages = [i for i, t in enumerate(layer_tokens) if len(t) >= 20]

        per_tool, doc_payload = {}, {"tools": {}}
        for tool in tools:
            record = outputs.get(doc.name, {}).get(tool)
            if record is None:
                continue
            ok = record["status"] == "ok"
            markdown = record.get("markdown") or ""
            out_tokens = tokens(to_plain(markdown))
            entry = {
                "status": record["status"],
                "error": record.get("error"),
                "seconds": record.get("seconds"),
                "structure": structure(markdown) if ok else None,
            }
            if ok:
                # Text-layer score over the pages that have a text layer
                ref = [t for i in text_pages for t in layer_tokens[i]]
                entry["layer"] = bag_scores(out_tokens, ref) if ref else None
                pages_md = record.get("pages")
                if pages_md and len(pages_md) == len(images):
                    entry["page_recall"] = [
                        bag_scores(tokens(to_plain(md)), layer_tokens[i])["recall"] if i in text_pages else None
                        for i, md in enumerate(pages_md)
                    ]
                    entry["empty_pages"] = sum(
                        1 for i, md in enumerate(pages_md) if i in text_pages and len(tokens(to_plain(md))) < 5
                    )
                if synthetic:
                    gt_tokens = tokens(to_plain(ground_truth))
                    counter = Counter(out_tokens)
                    by_source = {}
                    for check in checks:
                        by_source.setdefault(check["category"], []).append(snippet_recall(check["text"], counter))
                    known = set(gt_tokens) | {t for c in checks for t in tokens(c["text"])}
                    entry["gt"] = {
                        **{f"body_{k}": v for k, v in bag_scores(out_tokens, gt_tokens).items()},
                        "headings": heading_score(headings(ground_truth), headings(markdown)),
                        "tables": table_score(markdown_tables(ground_truth), markdown_tables(markdown)),
                        "order": order_score(gt_tokens, out_tokens),
                        "noise": (sum(1 for t in out_tokens if t not in known) / len(out_tokens)) if out_tokens else 0,
                        "sources": {k: statistics.mean(v) for k, v in by_source.items()},
                    }
            per_tool[tool] = entry
            doc_payload["tools"][tool] = {
                "markdown": markdown,
                "pages": record.get("pages") if record.get("pages") and len(record["pages"]) == len(images) else None,
                "status": record["status"],
                "error": record.get("error"),
            }

        doc_payload["reference"] = {
            "layer_pages": layer_texts,
            "ground_truth": ground_truth,
        }
        doc_payload["images"] = images
        doc_payload["id"] = did
        doc_payload["name"] = doc.name
        (out_dir / "data" / "docs" / f"{did}.js").write_text(
            f"window.__docData = window.__docData || {{}}; window.__docData[{json.dumps(did)}] = "
            + json.dumps(doc_payload, ensure_ascii=False)
            + ";"
        )
        doc_rows.append(
            {
                "id": did,
                "name": doc.name,
                "pages": len(images),
                "synthetic": synthetic,
                "text_layer_pages": len(text_pages),
                "size_mb": doc.stat().st_size / 1e6,
                "tools": per_tool,
            }
        )

    summary = {
        "tools": [{"id": t, "label": TOOLS[t], "setup": setups.get(t, {})} for t in tools],
        "docs": doc_rows,
        "sources": SOURCE_LABELS,
    }
    assessment = HERE / "assessment.json"
    if assessment.exists():
        summary["assessment"] = json.loads(assessment.read_text())
    (out_dir / "data" / "summary.js").write_text("window.__summary = " + json.dumps(summary, ensure_ascii=False) + ";")
    shutil.copy(HERE / "dashboard.html", out_dir / "index.html")
    print("dashboard:", out_dir / "index.html")


if __name__ == "__main__":
    main()
