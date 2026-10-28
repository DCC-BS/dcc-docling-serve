"""The private-use bullet patch for docling's list markers (plugins/list_bullets.py)."""

import importlib.util
from pathlib import Path

import pytest
from docling_core.types.doc import DoclingDocument

PLUGIN = Path(__file__).resolve().parents[2] / "plugins" / "list_bullets.py"


@pytest.fixture(scope="module")
def processor():
    spec = importlib.util.spec_from_file_location("dcc_list_bullets", PLUGIN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from docling.models.postprocessing.list_marker_processor import ListItemMarkerProcessor

    return ListItemMarkerProcessor()


def _list_item(text: str):
    doc = DoclingDocument(name="t")
    group = doc.add_list_group()
    return doc.add_list_item(text=text, parent=group)


@pytest.mark.parametrize(("char", "shown"), [("", "•"), ("", "▪"), ("", "✓")])
def test_private_use_bullet_split_off(processor, char, shown):
    item = processor.process_list_item(_list_item(f"{char} Transcribo: Dient der Umwandlung"))

    assert item.text == "Transcribo: Dient der Umwandlung"
    assert item.marker == shown
    assert item.enumerated is False


def test_markdown_has_no_private_use_character(processor):
    doc = DoclingDocument(name="t")
    group = doc.add_list_group()
    item = doc.add_list_item(text=" Bebbi Bot: Ermöglicht RAG-Bots", parent=group)
    processor.process_list_item(item)

    assert doc.export_to_markdown() == "- Bebbi Bot: Ermöglicht RAG-Bots"


def test_lone_private_use_bullet_is_a_marker(processor):
    assert processor._is_bullet_marker("")


def test_ordinary_bullets_unchanged(processor):
    item = processor.process_list_item(_list_item("• Erster Punkt"))

    assert (item.marker, item.text) == ("•", "Erster Punkt")


def test_patch_applied_once(processor):
    from docling.models.postprocessing.list_marker_processor import ListItemMarkerProcessor

    # A second import (two .pth paths, a reload) must not stack the patch
    spec = importlib.util.spec_from_file_location("dcc_list_bullets_again", PLUGIN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    patterns = ListItemMarkerProcessor()._bullet_patterns
    assert patterns.count(module._PATTERN) == 1
