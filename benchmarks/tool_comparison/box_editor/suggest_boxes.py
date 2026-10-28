# /// script
# requires-python = ">=3.12"
# dependencies = ["pypdfium2", "pillow"]
# ///
"""Suggest a box per text line for documents with a reference transcription, and build the box editor.

The suggestions come from the document's own text layer (for a scan: the scanner's OCR), never
from an engine under test: its words are grouped into lines, each word is corrected against the
reviewed transcription in the references file, and each line gets a category from its text and
position. The editor (editor.html) shows the page with these boxes; a person moves, resizes and
rotates them, corrects text and category, adds what the text layer missed and exports the result
as the box ground truth.

    uv run --script suggest_boxes.py --references ../internal/references.json \\
        --docs ../internal/docs --only <file.pdf> ... --out ../internal/boxes/editor.html
"""

from __future__ import annotations

import argparse
import base64
import difflib
import io
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pypdfium2 as pdfium

HERE = Path(__file__).resolve().parent
MAX_SIDE = 6000  # rendered page, in pixels
CATEGORIES = [
    "Title block",
    "Legend",
    "Header / footer",
    "View title",
    "Room",
    "Label",
    "Map label",
    "Level",
    "Dimension",
    "Window label",
    "Grid axis",
    "Key plan",
    "Scale bar",
    "Handwriting",
]


class Vocabulary(dict):
    """Normalised token -> the transcription's spelling, with how often each occurs."""

    counts: Counter


def norm(token: str) -> str:
    return re.sub(r"[^\w]", "", unicodedata.normalize("NFKC", token).lower())


def words(page: pdfium.PdfPage, height: float) -> list[dict]:
    """The text layer's words, top-left origin."""
    textpage = page.get_textpage()
    found = []
    for i in range(textpage.count_rects()):
        left, bottom, right, top = textpage.get_rect(i)
        # The bounded text can reach into the next line ("FQ \r\n2"): the first line is this word
        text = (textpage.get_text_bounded(left, bottom, right, top).strip().splitlines() or [""])[0].strip()
        if text:
            found.append({"text": text, "l": left, "t": height - top, "r": right, "b": height - bottom})
    return found


def lines(found: list[dict]) -> list[list[dict]]:
    """Words on one baseline and close together form a line; a wide gap starts a new one."""
    grouped: list[list[dict]] = []
    # Left to right: the last word of a line is its rightmost, and a superscript comes after its word
    for word in sorted(found, key=lambda w: (w["l"], w["t"])):
        h = max(word["b"] - word["t"], 1.0)
        for line in grouped:
            last = line[-1]
            lh = max(last["b"] - last["t"], 1.0)
            overlap = min(word["b"], last["b"]) - max(word["t"], last["t"])
            gap = word["l"] - last["r"]
            # A superscript ("m²", "3.68⁵") sits higher, right after its word with no space between
            raised = gap <= 0.6 * lh and abs((word["t"] + word["b"]) - (last["t"] + last["b"])) / 2 <= 0.8 * lh
            if (overlap >= 0.5 * min(h, lh) or raised) and -0.5 * h <= gap <= 1.5 * max(h, lh):
                word["tight"] = gap < 0.3 * lh
                line.append(word)
                break
        else:
            grouped.append([word])
    return grouped


def fix(token: str, vocabulary: Vocabulary) -> str | None:
    """The transcription's spelling of a text-layer token, or None for scanner noise."""
    key = norm(token)
    if not key:
        return None
    if key in vocabulary:
        return vocabulary[key]
    if len(key) <= 2:
        # A stroke or a speck read as a letter: only a two-letter word with the right first letter ("FO" for "FQ")
        close = [v for v in vocabulary if len(key) == len(v) == 2 and v[0] == key[0]]
        return vocabulary[max(close, key=lambda v: vocabulary.counts[v])] if close else None
    best = difflib.get_close_matches(key, list(vocabulary), n=1, cutoff=0.75 if len(key) > 3 else 0.6)
    return vocabulary[best[0]] if best else None


def align(keys: list[str], tokens: list[str], raw: str) -> tuple[str, float]:
    """The run of transcription words that best matches a line's corrected words, punctuation included.

    Words without letters or digits ("=", "/") are skipped while matching and kept in the
    answer between matched words, and at its ends where the line has them too ("NK = FQ =").
    """
    words_at = [i for i, t in enumerate(tokens) if norm(t)]
    keyed = [norm(tokens[i]) for i in words_at]
    best, score = (0, 0), 0.0
    n = len(keys)
    for size in dict.fromkeys((n, max(1, n - 1), n + 1)):  # the line's own length wins a tie
        for start in range(0, max(1, len(keyed) - size + 1)):
            window = keyed[start : start + size]
            if window[0] != keys[0] and window[-1] != keys[-1]:
                continue
            ratio = difflib.SequenceMatcher(None, keys, window, autojunk=False).ratio()
            if ratio > score:
                best, score = (start, start + size), ratio
    if not score:
        return "", 0.0
    first, last = words_at[best[0]], words_at[best[1] - 1]
    raw_tokens = raw.split()
    while first > 0 and not norm(tokens[first - 1]) and raw_tokens and not norm(raw_tokens[0]):
        first -= 1
    while last + 1 < len(tokens) and not norm(tokens[last + 1]) and raw_tokens and not norm(raw_tokens[-1]):
        last += 1
    return " ".join(tokens[first : last + 1]), score


def category(text: str, box: tuple[float, float, float, float], width: float, height: float, legend: set[str]) -> str:
    left, top, right, bottom = box
    cx, cy = (left + right) / 2 / width, (top + bottom) / 2 / height
    large = width > 1000  # the A1/A0 plans with a legend column
    if re.search(r"ANSICHT|QUERSCHNITT", text):
        return "View title"
    if large and cx > 0.79:
        if re.fullmatch(r"[A-FNS]", text):
            return "Key plan"
        return "Legend" if norm(text.split()[0]) in legend and cy < 0.45 else "Title block"
    if not large and cy > 0.84:
        return "Title block"
    if re.search(r"\b(FQ|NK)\b|^\d\.\d\d m", text):
        return "Window label"
    if "m.ü.M" in text or text.startswith("OK "):
        return "Level"
    if re.search(r"\b[AB]\.(OG|EG|UG)\d", text) or (text.isupper() and len(text) > 3):
        return "Room"
    if large and re.fullmatch(r"[A-F]|\d{1,2}", text):
        return "Grid axis"
    if re.fullmatch(r"[-±]?\d+([.,]\d+)?[⁰-⁹]*°?", text):
        return "Dimension"
    if not large:
        return "Header / footer" if cy < 0.12 else "Map label"
    return "Label"


def suggest(pdf: Path, reference: dict) -> dict:
    page = pdfium.PdfDocument(pdf)[0]
    width, height = page.get_size()
    scale = MAX_SIDE / max(width, height)
    image = page.render(scale=scale).to_pil().convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=82)
    transcription = " ".join(reference["parts"].values())
    vocabulary = Vocabulary((norm(t), t) for t in transcription.split() if norm(t))
    vocabulary.counts = Counter(norm(t) for t in transcription.split() if norm(t))
    legend = {norm(t) for t in reference["parts"].get("Legend", "").split()}
    tokens = transcription.split()
    instances = []
    for line in lines(words(page, height)):
        raw = " ".join(w["text"] for w in line)
        # A superscript belongs to the word before it: "m" + "2" is "m²", "3.68" + "5" is "3.68⁵"
        joined: list[str] = []
        for w in line:
            parts = w["text"].split()
            if w.get("tight") and joined and norm(joined[-1] + parts[0]) in vocabulary:
                joined[-1] += parts.pop(0)
            joined += parts
        fixed = [fix(t, vocabulary) for t in joined]
        keys = [norm(f) for f in fixed if f]
        if not keys:
            continue  # only scanner noise
        text, score = align(keys, tokens, raw)
        if score < 0.6:
            text = " ".join(f for f in fixed if f)
        box = (
            min(w["l"] for w in line),
            min(w["t"] for w in line),
            max(w["r"] for w in line),
            max(w["b"] for w in line),
        )
        changed = text != raw or score < 1.0
        instances.append(
            {
                "text": text,
                "category": category(text, box, width, height, legend),
                "cx": (box[0] + box[2]) / 2,
                "cy": (box[1] + box[3]) / 2,
                "w": box[2] - box[0],
                "h": box[3] - box[1],
                "angle": 0,
                "source": "text layer" + (f" (read: {raw})" if changed else ""),
                "flagged": changed or None in fixed,
                "reviewed": False,
            }
        )
    for i, it in enumerate(instances):
        it["sid"] = i  # stable: what a list of corrections refers to
    return {
        "file": pdf.name,
        "description": reference.get("description", ""),
        "pageWidth": width,
        "pageHeight": height,
        "imageWidth": image.width,
        "imageHeight": image.height,
        "image": "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode(),
        "parts": reference["parts"],
        "instances": instances,
    }


def apply(instances: list[dict], edits: dict) -> list[dict]:
    """Corrections from looking at the page: indices refer to the text-layer suggestions."""
    for i, fields in edits.get("set", {}).items():
        instances[int(i)].update(fields, source=instances[int(i)]["source"] + "; corrected by eye")
    gone = set(edits.get("delete", []))
    kept = [it for i, it in enumerate(instances) if i not in gone]
    added = [
        {"angle": 0, "flagged": False, "reviewed": False, "source": "placed by eye", **it}
        for it in edits.get("add", [])
    ]
    return kept + added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--only", nargs="+", required=True, help="document file names to annotate")
    parser.add_argument(
        "--edits", type=Path, help="JSON {file: {delete: [i], set: {i: fields}, add: [instance]}} from a visual review"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    references = json.loads(args.references.read_text())
    edits = json.loads(args.edits.read_text()) if args.edits else {}
    plans = []
    for name in args.only:
        plan = suggest(args.docs / name, references[name])
        plan["instances"] = apply(plan["instances"], edits.get(name, {}))
        plans.append(plan)
        print(name, len(plan["instances"]), "boxes,", sum(i["flagged"] for i in plan["instances"]), "flagged")
    data = json.dumps({"categories": CATEGORIES, "plans": plans}, ensure_ascii=False)
    template = (HERE / "editor.html").read_text()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(template.replace("/*DATA*/null", data))
    print("editor:", args.out)


if __name__ == "__main__":
    main()
