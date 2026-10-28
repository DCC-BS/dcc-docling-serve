# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx"]
# ///
"""Convert the documents with the hosted Jina Reader API (uploads every document to r.jina.ai).

Private documents (benchmarks/internal) must not leave the machine, so by default only the
synthetic set is uploaded; --docs sends other folders.

The API key is read from the jina_key entry of the .env file in the repository parent folder,
or from the JINA_KEY environment variable.

    uv run --script run_jina.py
"""

import os
import re

import httpx
from common import HERE, INTERNAL_DOCS, Recorder, list_docs, parser


def api_key() -> str:
    if key := os.environ.get("JINA_KEY"):
        return key
    for env_file in (HERE.parents[2] / ".env", HERE.parents[1] / ".env"):
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                match = re.match(r"\s*jina_key\s*=\s*(.+)", line, re.IGNORECASE)
                if match:
                    return match.group(1).strip().strip("\"'")
    raise SystemExit("No jina_key found in .env and JINA_KEY is not set")


def main() -> None:
    cli = parser(__doc__)
    cli.set_defaults(docs=[HERE / "synthetic"])
    args = cli.parse_args()
    if any(INTERNAL_DOCS.resolve() in (d.resolve(), *d.resolve().parents) for d in args.docs):
        raise SystemExit(f"{INTERNAL_DOCS} holds private documents; they are not uploaded to Jina")
    recorder = Recorder("jina", args.results, args.force)
    docs = list_docs(args.docs, args.only)
    client = httpx.Client(
        timeout=httpx.Timeout(900, connect=30),
        headers={
            "Authorization": f"Bearer {api_key()}",
            "Accept": "application/json",
            "X-No-Cache": "true",
            "X-Timeout": "180",
        },  # fmt: skip
    )
    recorder.setup(model_load_s=0.0, endpoint="https://r.jina.ai/")

    def convert(doc):
        with doc.open("rb") as handle:
            response = client.post("https://r.jina.ai/", files={"file": (doc.name, handle)})
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
        data = response.json().get("data") or {}
        return {
            "markdown": data.get("content") or "",
            "pages": None,
            "meta": {"numPages": data.get("numPages"), "usage": data.get("usage")},
        }

    recorder.run(docs, convert)


if __name__ == "__main__":
    main()
