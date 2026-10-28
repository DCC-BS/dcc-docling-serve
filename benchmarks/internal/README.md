# Private benchmark data

This folder holds documents and ground truth that must not leave your machine: internal
documents, their transcriptions, and anything with personal data. Git ignores everything
here except this README, and a pre-commit hook rejects any file under `benchmarks/internal/`
that is added with `git add -f`. The debug image build leaves it out of its Docker context.

The benchmarks run without this folder. Then they use the synthetic documents, the
handwritten note `data/ocr.png` and the public XFUND forms (`tool_comparison/fetch_public.py`).

## Layout

```
benchmarks/internal/
├── docs/              # documents the benchmarks convert and score
├── sensitive/         # documents that are never converted by a benchmark (kept for manual checks)
├── references.json    # transcriptions of documents in docs/ (format below)
└── boxes/             # text boxes per document: the box ground truth and its editor
```

`DCC_INTERNAL_DOCS=/other/path` points the benchmarks at another documents folder.
Only the top level of `docs/` is scanned; subfolders are left alone.

## Ground truth: what to add for a new document

How a document is scored depends on what you give it. Put the document into `docs/`, then:

| Document | Add | Scores |
| --- | --- | --- |
| PDF with a correct text layer | nothing | word recall and precision against the text layer, per page |
| Scan, photo, or PDF whose text layer is wrong or incomplete (scanner OCR, plans) | an entry in `references.json` | word recall and precision against the transcription, and recall per part |
| Document where structure matters | `<name>.gt.md`, optionally `<name>.checks.json`, next to it in `docs/` | body F1, headings, tables, reading order, noise, recall per source |

A scanner's text layer is OCR, not ground truth: on the plans it misses rotated text and
garbles numbers ("1.36 m²" as "136 m2"). Score such documents against a transcription.

**`references.json`.** One entry per file name; the parts are scored separately:

```json
{
  "note.jpg": {
    "description": "Phone photo of handwritten notes on a printed notepad",
    "parts": {"Print": "ALTES SPITAL", "Handwriting": "Strategie ↗ Effizienz MA …"}
  }
}
```

Write down every visible word, repeated labels as often as they appear, and nothing that is
redacted. Order and line breaks don't matter: the score compares the bags of words. Left-out
text is not neutral, because it can't be missed: the engines that are weak on it look better.

**`<name>.gt.md`.** The exact markdown a perfect conversion would give: every word, `#`
headings, tables as markdown tables, in reading order. Start from the best engine's output
and correct it by hand; a person has to check every line, because a ground truth drafted
from one engine's output favours that engine. `<name>.checks.json` lists snippets with a
category (`[{"category": "body", "page": 1, "text": "…"}]`) for the recall per source.

## Box ground truth

`boxes/` holds where each line of text stands on the page, for the text-and-position eval.
`tool_comparison/box_editor/suggest_boxes.py` builds the editor from the documents, their
transcriptions in `references.json` and a list of corrections (`boxes/vision_edits.json`):

```bash
cd benchmarks/tool_comparison/box_editor
uv run --script suggest_boxes.py --references ../../internal/references.json \
    --docs ../../internal/docs --edits ../../internal/boxes/vision_edits.json \
    --only <file.pdf> ... --out ../../internal/boxes/editor.html
```

Open `boxes/editor.html` in a browser. Move, resize and rotate boxes with the mouse, correct
text and category, add what is missing (the coverage panel lists the transcription words
without a box), then **Export all**. The export (`boxes.json`, PDF points with a top-left
origin, four corners per box) is the box ground truth; keep it as `boxes/boxes.json`. Edits
are kept in the browser until you export, and **Import** loads an export back in.
