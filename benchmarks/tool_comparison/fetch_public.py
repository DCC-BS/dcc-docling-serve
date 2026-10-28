# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Download public test documents with ground truth into public/ (git-ignored).

XFUND German validation split: 50 scanned and photographed forms (applications, registrations,
questionnaires), many filled in by hand, with word-level transcriptions. Licence CC BY-NC-SA 4.0
(https://github.com/doc-analysis/XFUND): fine for internal, non-commercial evaluation; do not
redistribute the images or outputs made from them.

Writes public/xfund_de/<form>.jpg and public/xfund_de.references.json, which engines.config.json
reads as reference transcriptions (all words of a form, top to bottom).

    uv run --script benchmarks/tool_comparison/fetch_public.py
"""

import io
import json
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELEASE = "https://github.com/doc-analysis/XFUND/releases/download/v1.0"


def download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=300) as response:  # noqa: S310 -- pinned https release URL
        return response.read()


def main() -> None:
    out = HERE / "public" / "xfund_de"
    out.mkdir(parents=True, exist_ok=True)
    labels = json.loads(download(f"{RELEASE}/de.val.json"))
    with zipfile.ZipFile(io.BytesIO(download(f"{RELEASE}/de.val.zip"))) as archive:
        for member in archive.namelist():
            if member.lower().endswith((".jpg", ".png")):
                (out / Path(member).name).write_bytes(archive.read(member))
    references = {}
    for form in labels["documents"]:
        entities = sorted(form["document"], key=lambda e: (e["box"][1] // 25, e["box"][0]))
        words = " ".join(e["text"].strip() for e in entities if e["text"].strip())
        references[form["img"]["fname"]] = {
            "description": "XFUND German form (CC BY-NC-SA 4.0), word-level transcription",
            "parts": {"Form": words},
        }
    (HERE / "public" / "xfund_de.references.json").write_text(json.dumps(references, ensure_ascii=False, indent=1))
    print(f"wrote {len(references)} forms to {out}")


if __name__ == "__main__":
    main()
