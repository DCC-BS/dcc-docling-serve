"""Shared fixtures for end-to-end tests."""

import os
import socket
from pathlib import Path
from urllib.parse import urlparse

import pytest

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

DEFAULT_DOCLING_SERVE_URL = os.environ.get(
    "DOCLING_SERVE_URL",
    f"http://localhost:{os.environ.get('DOCLING_HOST_PORT', '5001')}",
)


def _is_url_reachable(url: str, timeout: float = 1.0) -> bool:
    """Check if target host and port are accepting connections."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, ValueError):
        return False


def pytest_collection_modifyitems(config, items):
    """Skip e2e tests when docling-serve is not reachable."""
    url = os.environ.get("DOCLING_SERVE_URL", DEFAULT_DOCLING_SERVE_URL)
    if url and _is_url_reachable(url):
        return
    skip = pytest.mark.skip(reason=f"docling-serve not reachable at {url}")
    for item in items:
        if "e2e" in item.keywords:
            item.add_marker(skip)


@pytest.fixture()
def docling_serve_url():
    return os.environ.get("DOCLING_SERVE_URL", DEFAULT_DOCLING_SERVE_URL).rstrip("/")


@pytest.fixture()
def ocr_image_path():
    return DATA_DIR / "ocr.png"


@pytest.fixture()
def ocr_no_text_image_path():
    return DATA_DIR / "ocr_no_text.jpg"


@pytest.fixture()
def legacy_doc_path():
    """A Word 97 (.doc, OLE2) file: docling converts it to .docx with LibreOffice first."""
    return DATA_DIR / "legacy_word97.doc"
