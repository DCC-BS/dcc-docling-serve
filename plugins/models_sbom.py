"""Write a CycloneDX list of the models this image adds on top of docling-serve.

    python models_sbom.py OUT.cdx.json REPO@COMMIT=DIR [REPO@COMMIT=DIR ...]

Each model is a Hugging Face repository downloaded at a full commit hash into DIR. The list
names the repository and commit (pkg:huggingface purl) and the SHA-256 of each file in DIR
that the commit holds; files it does not hold, such as the keys file docling-pp-ocrv6 derives
from inference.yml, are left out. The form is the one ``docling-tools models download`` uses
for models.cdx.json (docling-project/docling#5489). SBOM tools that read SBOM files inside an
image (syft, used by ``docker build --sbom=true``) then list these models too.
"""

import hashlib
import json
import re
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

COMMIT = re.compile(r"^[0-9a-f]{40}$")
REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse(arg: str) -> tuple[str, str, Path]:
    """REPO@COMMIT=DIR; stops on a moving revision or a malformed argument."""
    spec, sep, directory = arg.partition("=")
    repo, at, commit = spec.partition("@")
    if not (sep and at and REPO.match(repo) and COMMIT.match(commit)):
        sys.exit(f"{arg!r}: expected REPO@COMMIT=DIR with a full 40-character commit hash")
    return repo, commit, Path(directory)


def _hub_files(repo: str, commit: str) -> list[str]:
    from huggingface_hub import HfApi

    return HfApi().list_repo_files(repo, revision=commit)


def _component(repo: str, commit: str, directory: Path, repo_files: Iterable[str]) -> dict:
    """The model at ``commit``, with the files of the commit found in ``directory``."""
    files = sorted(name for name in set(repo_files) if (directory / name).is_file())
    if not files:
        sys.exit(f"{repo}@{commit}: none of the commit's files are in {directory}")
    purl = f"pkg:huggingface/{repo}@{commit}"
    return {
        "type": "machine-learning-model",
        "bom-ref": purl,
        "name": repo,
        "version": commit,
        "purl": purl,
        "externalReferences": [{"type": "distribution", "url": f"https://huggingface.co/{repo}/tree/{commit}"}],
        "components": [
            {
                "type": "file",
                "name": name,
                "hashes": [{"alg": "SHA-256", "content": _sha256(directory / name)}],
            }
            for name in files
        ],
    }


def main(argv: list[str], list_files: Callable[[str, str], Iterable[str]] = _hub_files) -> None:
    if len(argv) < 2:
        sys.exit(__doc__)
    out, models = Path(argv[0]), sorted(_parse(arg) for arg in argv[1:])
    bom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "version": 1,
        "metadata": {"tools": {"components": [{"type": "application", "name": "dcc-docling-serve models_sbom.py"}]}},
        "components": [
            _component(repo, commit, directory, list_files(repo, commit)) for repo, commit, directory in models
        ],
    }
    out.write_text(json.dumps(bom, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
