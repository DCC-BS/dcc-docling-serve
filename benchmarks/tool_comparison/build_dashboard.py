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
  - documents listed in the references file (images without a text layer): against a
    hand-made transcription.

What is compared comes from a JSON config (paths relative to the config file):
    title, heading, noun         page title, heading, what the tools are called ("converters")
    tool_noun                    one of them ("tool")
    tools                        [{id, label, engine?, layout?}] in display order
    results, docs, output        results folder, document folders or files, dashboard folder
                                 (default <results>/dashboard)
    template, assessment         HTML page and pros/cons file shown on the overview
    references                   {file name: {description, parts: {name: text}}} (optional)
    time                         "wall" (record seconds) or "server" (meta.server_processing_s)
    compare                      {left, right}: tools selected in the Compare tab
    matrix                       tool ids with engine/layout, shown as an engine x layout grid
    pairs                        [{label, released, fixed}]: plugin versions to compare
    extra_words                  list the output words that are not in the ground truth/reference
    method                       HTML bullet points for the Method tab

Output: <output>/index.html plus data/ and pages/ next to it.

    uv run --script build_dashboard.py                                   # converter comparison
    uv run --script build_dashboard.py --config engines.config.json --results <engine results>
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
from common import HERE, list_docs
from PIL import Image

DEFAULT_CONFIG = HERE / "converters.config.json"
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


# ------------------------------------------------------------------ config
def load_config(args: argparse.Namespace) -> dict:
    """Read the JSON config, resolve its paths (relative to the config file) and apply CLI overrides."""
    config = json.loads(args.config.read_text())
    base = args.config.resolve().parent

    def resolve(value):
        return value if value is None else (base / value).resolve()

    config["results"] = args.results or resolve(config["results"])
    config["docs"] = args.docs or [resolve(d) for d in config["docs"]]
    config["output"] = args.output or resolve(config.get("output")) or config["results"] / "dashboard"
    config["template"] = resolve(config.get("template", "dashboard.html"))
    for key in ("assessment", "references"):
        config[key] = resolve(config.get(key))
    config["tools"] = {t["id"]: t for t in config["tools"]}
    return config


def display_config(config: dict) -> dict:
    """The part of the config the page needs."""
    keys = ("title", "heading", "noun", "tool_noun", "time", "compare", "matrix", "pairs", "extra_words", "method")
    shown = {k: config[k] for k in keys if k in config}
    if "matrix" in shown:  # labels for cells whose tool has no results yet
        shown["matrix"] = [config["tools"][t] for t in shown["matrix"]]
    return shown


# ------------------------------------------------------------------ documents
def doc_id(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def load_outputs(results: Path, tools: dict) -> dict[str, dict[str, dict]]:
    outputs: dict[str, dict[str, dict]] = {}
    for tool in tools:
        folder = results / "outputs" / tool
        if not folder.exists():
            continue
        for path in folder.glob("*.json"):
            if path.name == "_setup.json":
                continue
            try:
                record = json.loads(path.read_text())
            except json.JSONDecodeError:  # still being written by a running runner
                print("skipping unreadable", path, flush=True)
                continue
            outputs.setdefault(record["doc"], {})[tool] = record
    return outputs


def load_setups(results: Path, tools: dict) -> dict[str, dict]:
    setups = {}
    for tool in tools:
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


def align_pages(chunks: list[str], page_refs: list[Counter]) -> list[str]:
    """Place k page chunks on n > k pages, in order, maximising word overlap with each page's reference.

    docling leaves pages without any output out of the page-break split, so a shorter split means
    some pages came out empty; this finds which ones.
    """
    k, n = len(chunks), len(page_refs)
    chunk_words = [Counter(tokens(to_plain(c))) for c in chunks]
    score = [[sum((chunk_words[i] & page_refs[j]).values()) for j in range(n)] for i in range(k)]
    worst = float("-inf")
    best = [[0.0] * (n + 1)] + [[worst] * (n + 1) for _ in range(k)]
    for i in range(1, k + 1):
        for j in range(i, n + 1):
            best[i][j] = max(best[i][j - 1], best[i - 1][j - 1] + score[i - 1][j - 1])
    pages, i = [""] * n, k
    for j in range(n, 0, -1):
        if i == 0:
            break
        if i < j and best[i][j] == best[i][j - 1]:
            continue  # page j stays empty
        pages[j - 1] = chunks[i - 1]
        i -= 1
    return pages


def closest_words(words: list[str], vocabulary: list[str]) -> dict[str, str | None]:
    return {w: next(iter(difflib.get_close_matches(w, vocabulary, n=1, cutoff=0.7)), None) for w in words}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--docs", type=Path, nargs="+", help="Override the config's document folders/files")
    parser.add_argument("--results", type=Path, help="Override the config's results folder")
    parser.add_argument("--output", type=Path, help="Dashboard folder (default <results>/dashboard)")
    args = parser.parse_args()
    config = load_config(args)
    tool_config = config["tools"]
    server_time = config.get("time") == "server"

    out_dir = config["output"]
    (out_dir / "data" / "docs").mkdir(parents=True, exist_ok=True)
    outputs, setups = load_outputs(config["results"], tool_config), load_setups(config["results"], tool_config)
    tools = [t for t in tool_config if t in setups or any(t in o for o in outputs.values())]
    references = json.loads(config["references"].read_text()) if config["references"] else {}
    synthetic_dir = HERE / "synthetic"

    doc_rows = []
    for doc in list_docs(config["docs"]):
        print("scoring", doc.name, flush=True)
        did = doc_id(doc.name)
        images, layer_texts = render_pages(doc, out_dir / "pages" / did)
        gt_path = synthetic_dir / f"{doc.stem}.gt.md"
        synthetic = gt_path.exists() and doc.parent.resolve() == synthetic_dir.resolve()
        ground_truth = gt_path.read_text() if synthetic else None
        checks = json.loads((synthetic_dir / f"{doc.stem}.checks.json").read_text()) if synthetic else []
        reference = references.get(doc.name)
        reference_md = (
            "\n\n".join(f"## {name}\n\n{text}" for name, text in reference["parts"].items()) if reference else None
        )
        layer_tokens = [tokens(t) for t in layer_texts]
        text_pages = [i for i, t in enumerate(layer_tokens) if len(t) >= 20]

        # Per-page markdown. A shorter split (pages without output are left out) is aligned to the pages.
        records = {t: outputs.get(doc.name, {}).get(t) for t in tools}
        records = {t: r for t, r in records.items() if r is not None}
        page_splits, aligned = {}, set()
        for tool, record in records.items():
            if record["status"] == "ok" and record.get("pages") and len(record["pages"]) == len(images):
                page_splits[tool] = record["pages"]
        page_words = {t: [Counter(tokens(to_plain(md))) for md in pages] for t, pages in page_splits.items()}
        page_refs = [
            Counter(layer_tokens[i])
            if i in text_pages
            else max((w[i] for w in page_words.values()), key=lambda c: sum(c.values()), default=Counter())
            for i in range(len(images))
        ]
        for tool, record in records.items():
            pages_md = record.get("pages")
            if record["status"] == "ok" and pages_md and len(pages_md) < len(images):
                page_splits[tool] = align_pages(pages_md, page_refs)
                page_words[tool] = [Counter(tokens(to_plain(md))) for md in page_splits[tool]]
                aligned.add(tool)
        # Pages with text: a text layer, or any tool read at least 20 words from it (scans, photos).
        page_kind = [
            "layer"
            if i in text_pages
            else "scan"
            if any(sum(w[i].values()) >= 20 for w in page_words.values()) or (reference and len(images) == 1)
            else None
            for i in range(len(images))
        ]

        per_tool, doc_payload = {}, {"tools": {}}
        for tool, record in records.items():
            ok = record["status"] == "ok"
            markdown = record.get("markdown") or ""
            out_tokens = tokens(to_plain(markdown))
            seconds = record.get("seconds")
            if server_time and (record.get("meta") or {}).get("server_processing_s") is not None:
                seconds = record["meta"]["server_processing_s"]
            entry = {
                "status": record["status"],
                "error": record.get("error"),
                "seconds": seconds,
                "structure": structure(markdown) if ok else None,
            }
            if server_time:
                entry["wall_seconds"] = record.get("seconds")
            pages_md = page_splits.get(tool)
            if ok:
                # Text-layer score over the pages that have a text layer
                ref = [t for i in text_pages for t in layer_tokens[i]]
                entry["layer"] = bag_scores(out_tokens, ref) if ref else None
                if pages_md:
                    entry["page_recall"] = [
                        bag_scores(tokens(to_plain(md)), layer_tokens[i])["recall"] if i in text_pages else None
                        for i, md in enumerate(pages_md)
                    ]
                    entry["empty_pages"] = sum(
                        1 for i, md in enumerate(pages_md) if i in text_pages and len(tokens(to_plain(md))) < 5
                    )
                if server_time or config.get("pairs"):
                    entry["words"] = len(out_tokens)
                    if record.get("pages"):
                        entry["page_split"] = len(record["pages"])
                    if pages_md:
                        empty = [i for i, w in enumerate(page_words[tool]) if page_kind[i] and sum(w.values()) < 5]
                        entry["empty_text_pages"] = {
                            kind: [i + 1 for i in empty if page_kind[i] == kind] for kind in ("layer", "scan")
                        }
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
                if reference:
                    ref_tokens = tokens(reference_md)
                    counter = Counter(out_tokens)
                    entry["ref"] = {
                        **bag_scores(out_tokens, [t for text in reference["parts"].values() for t in tokens(text)]),
                        "parts": {name: snippet_recall(text, counter) for name, text in reference["parts"].items()},
                    }
                    known = set(ref_tokens)
                if config.get("extra_words") and (synthetic or reference):
                    extra = Counter(t for t in out_tokens if t not in known)
                    closest = closest_words(list(extra), sorted(known))
                    entry["extra_words"] = {
                        "total": len(out_tokens),
                        "extra": sum(extra.values()),
                        "words": [[w, n, closest[w]] for w, n in extra.most_common(300)],
                    }
            per_tool[tool] = entry
            doc_payload["tools"][tool] = {
                "markdown": markdown,
                "pages": pages_md,
                "status": record["status"],
                "error": record.get("error"),
            }
            if tool in aligned:
                doc_payload["tools"][tool]["pages_aligned"] = len(record["pages"])

        doc_payload["reference"] = {
            "layer_pages": layer_texts,
            "ground_truth": ground_truth,
        }
        if reference:
            doc_payload["reference"]["transcription"] = reference_md
        doc_payload["images"] = images
        doc_payload["id"] = did
        doc_payload["name"] = doc.name
        (out_dir / "data" / "docs" / f"{did}.js").write_text(
            f"window.__docData = window.__docData || {{}}; window.__docData[{json.dumps(did)}] = "
            + json.dumps(doc_payload, ensure_ascii=False)
            + ";"
        )
        row = {
            "id": did,
            "name": doc.name,
            "pages": len(images),
            "synthetic": synthetic,
            "text_layer_pages": len(text_pages),
            "size_mb": doc.stat().st_size / 1e6,
            "tools": per_tool,
        }
        if reference:
            row["reference"] = {"description": reference.get("description"), "parts": list(reference["parts"])}
        if server_time or config.get("pairs"):
            row["scan_text_pages"] = page_kind.count("scan")
        doc_rows.append(row)

    summary = {
        "tools": [
            {"id": t, "label": tool_config[t]["label"], "setup": setups.get(t, {})}
            | {k: v for k, v in tool_config[t].items() if k not in ("id", "label")}
            for t in tools
        ],
        "docs": doc_rows,
        "sources": SOURCE_LABELS,
        "config": display_config(config),
    }
    if config["assessment"] and config["assessment"].exists():
        summary["assessment"] = json.loads(config["assessment"].read_text())
    (out_dir / "data" / "summary.js").write_text("window.__summary = " + json.dumps(summary, ensure_ascii=False) + ";")
    shutil.copy(config["template"], out_dir / "index.html")
    print("dashboard:", out_dir / "index.html")


if __name__ == "__main__":
    main()
