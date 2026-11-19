"""End-to-end tests against a running docling-serve + vLLM stack.

Set DOCLING_SERVE_URL (e.g. http://localhost:5001) to run these tests.
"""

import base64
from pathlib import Path

import httpx
import pytest

TIMEOUT = 300  # generous timeout for model inference


def _encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _convert_file(
    base_url: str,
    image_path: Path,
    *,
    ocr_engine: str | None = None,
    layout_kind: str | None = None,
) -> dict:
    """Submit a single image for conversion and return the full response body."""
    options: dict = {
        "to_formats": ["md"],
        "image_export_mode": "placeholder",
        "ocr": True,
        "force_ocr": True,
    }
    if ocr_engine:
        options["ocr_engine"] = ocr_engine
    if layout_kind:
        options["layout_custom_config"] = {"kind": layout_kind}

    payload = {
        "sources": [
            {
                "kind": "file",
                "base64_string": _encode_image(image_path),
                "filename": image_path.name,
            }
        ],
        "options": options,
    }

    response = httpx.post(
        f"{base_url}/v1/convert/source",
        json=payload,
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def _extract_markdown(body: dict) -> str:
    return body.get("document", {}).get("md_content", "")


# ── GLM-OCR tests ──────────────────────────────────────────────────────


@pytest.mark.e2e
def test_glm_ocr_returns_text(docling_serve_url, ocr_image_path):
    """Image with text should produce non-empty markdown via GLM-OCR."""
    body = _convert_file(docling_serve_url, ocr_image_path, ocr_engine="glm-ocr-remote")
    md = _extract_markdown(body)
    assert len(md.strip()) > 0, "Expected non-empty markdown for image with text"


@pytest.mark.e2e
def test_glm_ocr_no_text_image(docling_serve_url, ocr_no_text_image_path):
    """Image without text should produce minimal/empty markdown via GLM-OCR."""
    body = _convert_file(docling_serve_url, ocr_no_text_image_path, ocr_engine="glm-ocr-remote")
    md = _extract_markdown(body)
    assert len(md.strip()) < 1000, f"Expected minimal text or caption, got {len(md.strip())} chars"


# ── PP-DocLayout-V3 + GLM-OCR combined tests ───────────────────────────


@pytest.mark.e2e
def test_ppdoclayout_with_glm_ocr(docling_serve_url, ocr_image_path):
    """Both plugins active: PP-DocLayout-V3 layout + GLM-OCR OCR."""
    body = _convert_file(
        docling_serve_url,
        ocr_image_path,
        ocr_engine="glm-ocr-remote",
        layout_kind="ppdoclayout-v3",
    )
    md = _extract_markdown(body)
    assert len(md.strip()) > 0, "Expected non-empty markdown with both plugins"


@pytest.mark.e2e
def test_ppdoclayout_with_glm_ocr_no_text(docling_serve_url, ocr_no_text_image_path):
    """Both plugins active on a no-text image."""
    body = _convert_file(
        docling_serve_url,
        ocr_no_text_image_path,
        ocr_engine="glm-ocr-remote",
        layout_kind="ppdoclayout-v3",
    )
    md = _extract_markdown(body)
    assert len(md.strip()) < 1000, f"Expected minimal text or caption, got {len(md.strip())} chars"


# ── Default engine tests ───────────────────────────────────────────────


@pytest.mark.e2e
def test_default_engines(docling_serve_url, ocr_image_path):
    """Conversion with default engines (no custom OCR/layout) should succeed."""
    body = _convert_file(docling_serve_url, ocr_image_path)
    md = _extract_markdown(body)
    assert len(md.strip()) > 0, "Expected non-empty markdown with default engines"


# ── PP-DocLayout-V3 only (default OCR) ─────────────────────────────────


@pytest.mark.e2e
def test_ppdoclayout_default_ocr(docling_serve_url, ocr_image_path):
    """PP-DocLayout-V3 layout with default OCR engine."""
    body = _convert_file(
        docling_serve_url,
        ocr_image_path,
        layout_kind="ppdoclayout-v3",
    )
    md = _extract_markdown(body)
    assert len(md.strip()) > 0, "Expected non-empty markdown with PP-DocLayout-V3 layout"


# ── Legacy Office formats (LibreOffice in the image) ──────────────────


@pytest.mark.e2e
def test_legacy_word_doc(docling_serve_url, legacy_doc_path):
    """A Word 97 .doc converts: without soffice in the image docling fails to open it."""
    payload = {
        "sources": [
            {
                "kind": "file",
                "base64_string": base64.b64encode(legacy_doc_path.read_bytes()).decode("utf-8"),
                "filename": legacy_doc_path.name,
            }
        ],
        "options": {"to_formats": ["md"]},
    }
    response = httpx.post(f"{docling_serve_url}/v1/convert/source", json=payload, timeout=TIMEOUT)
    response.raise_for_status()
    body = response.json()
    assert body["status"] == "success", body.get("errors")
    md = _extract_markdown(body)
    assert "Wegleitung Testdokument" in md
    assert "Punkt zwei" in md
