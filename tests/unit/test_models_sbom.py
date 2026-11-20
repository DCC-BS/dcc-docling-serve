"""The list of baked-in models for the image SBOM (plugins/models_sbom.py)."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "plugins" / "models_sbom.py"
COMMIT = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture(scope="module")
def models_sbom():
    spec = importlib.util.spec_from_file_location("dcc_models_sbom", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lists_commit_and_digests_of_the_commits_files_only(tmp_path, models_sbom):
    rec = tmp_path / "rec"
    rec.mkdir()
    (rec / "inference.onnx").write_bytes(b"weights")
    (rec / "inference.yml").write_bytes(b"config")
    (rec / "ppocrv6_keys.txt").write_bytes(b"derived at build time")
    out = tmp_path / "dcc-models.cdx.json"

    hub = {"org/rec": ["inference.onnx", "inference.yml", "README.md", "other.onnx"]}
    models_sbom.main([str(out), f"org/rec@{COMMIT}={rec}"], lambda repo, _: hub[repo])

    bom = json.loads(out.read_text())
    assert bom["bomFormat"] == "CycloneDX"
    (model,) = bom["components"]
    assert model["type"] == "machine-learning-model"
    assert model["purl"] == f"pkg:huggingface/org/rec@{COMMIT}"
    assert {f["name"]: f["hashes"][0]["content"] for f in model["components"]} == {
        "inference.onnx": hashlib.sha256(b"weights").hexdigest(),
        "inference.yml": hashlib.sha256(b"config").hexdigest(),
    }


@pytest.mark.parametrize(
    "arg",
    [
        "org/rec@main=/x",  # moving revision
        f"org/rec@{COMMIT[:12]}=/x",  # short hash
        f"org/rec@{COMMIT}",  # no directory
        f"rec@{COMMIT}=/x",  # no organisation
    ],
)
def test_stops_on_a_malformed_argument(tmp_path, models_sbom, arg):
    with pytest.raises(SystemExit):
        models_sbom.main([str(tmp_path / "out.json"), arg], lambda *_: [])


def test_stops_when_none_of_the_commits_files_were_downloaded(tmp_path, models_sbom):
    with pytest.raises(SystemExit, match="none of the commit's files"):
        models_sbom.main(
            [str(tmp_path / "out.json"), f"org/rec@{COMMIT}={tmp_path}"],
            lambda *_: ["inference.onnx"],
        )
