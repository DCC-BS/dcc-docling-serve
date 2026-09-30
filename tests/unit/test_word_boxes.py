"""Unit tests for plugins/word_boxes.py: the words RapidOCR reads."""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from docling.datamodel.pipeline_options import EasyOcrOptions, RapidOcrOptions
from docling_core.types.doc import CoordOrigin
from docling_core.types.doc.page import BoundingRectangle, PdfCellRenderingMode, PdfTextCell, TextCell

PATCH = Path(__file__).parents[2] / "plugins" / "word_boxes.py"


@pytest.fixture(scope="module")
def wb():
    """The patch module, loaded without hooking docling in this process."""
    os.environ["DCC_WORD_BOXES"] = "0"
    spec = importlib.util.spec_from_file_location("dcc_word_boxes", PATCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rect(left, top, right, bottom):
    return BoundingRectangle(
        r_x0=left,
        r_y0=top,
        r_x1=right,
        r_y1=top,
        r_x2=right,
        r_y2=bottom,
        r_x3=left,
        r_y3=bottom,
        coord_origin=CoordOrigin.TOPLEFT,
    )


def text_layer_word(text, box, invisible=False):
    mode = PdfCellRenderingMode.INVISIBLE if invisible else PdfCellRenderingMode.FILL_TEXT
    return PdfTextCell(
        index=0,
        text=text,
        orig=text,
        rect=rect(*box),
        from_ocr=False,
        rendering_mode=mode,
        text_direction="left_to_right",
        font_key="f",
        font_name="f",
        widget=False,
    )


def ocr_word(text, box):
    return TextCell(index=0, text=text, orig=text, rect=rect(*box), from_ocr=True, confidence=0.9)


def page(*words):
    return SimpleNamespace(
        size=SimpleNamespace(width=600, height=800), parsed_page=SimpleNamespace(word_cells=list(words))
    )


class TestOptions:
    def test_rapidocr_is_asked_for_word_boxes(self, wb):
        options = RapidOcrOptions(rapidocr_params={"Global.text_score": 0.6})
        asked = wb._ocr_for_word_boxes(options)
        assert asked.rapidocr_params == {"Global.text_score": 0.6, "Global.return_word_box": True}
        assert options.rapidocr_params == {"Global.text_score": 0.6}

    def test_other_engines_are_left_alone(self, wb):
        options = EasyOcrOptions()
        assert wb._ocr_for_word_boxes(options) is options

    def test_only_a_model_asked_for_words_reads_them(self, wb):
        asked = wb._ocr_for_word_boxes(RapidOcrOptions())
        assert wb._reads_words(SimpleNamespace(options=asked)) is True
        assert wb._reads_words(SimpleNamespace(options=RapidOcrOptions())) is False


class TestOcrWords:
    def test_words_are_placed_in_page_points(self, wb):
        results = [
            [
                ("Anna", 0.9, [[30, 60], [120, 60], [120, 90], [30, 90]]),
                ("Muster", 0.8, [[135, 60], [300, 60], [300, 90], [135, 90]]),
            ]
        ]
        words = wb._ocr_words(results, scale=3)
        assert [w.text for w in words] == ["Anna", "Muster"]
        assert all(w.from_ocr for w in words)
        box = words[1].rect.to_bounding_box()
        assert (box.l, box.t, box.r, box.b) == (45, 20, 100, 30)

    def test_empty_and_unplaced_words_are_skipped(self, wb):
        # what RapidOCR answers for a picture it read no text in
        results = [[()], [("lost", 0.7, None), ("  ", 0.7, [[0, 0], [3, 0], [3, 3], [0, 3]])]]
        assert wb._ocr_words(results, scale=3) == []

    def test_a_page_read_empty(self, wb):
        # what RapidOCR answers for a page it read nothing on
        assert wb._ocr_words((("", 1.0, None),), scale=3) == []

    def test_no_results(self, wb):
        assert wb._ocr_words(None, scale=3) == []


class TestNotDrawn:
    def test_a_word_the_text_layer_draws_is_not_doubled(self, wb):
        drawn = text_layer_word("Anna", (100, 102, 140, 112))
        read = ocr_word("Anna", (98, 98, 142, 116))
        assert wb._not_drawn([read], page(drawn)) == []

    def test_a_word_only_the_pixels_show_is_kept(self, wb):
        drawn = text_layer_word("Anna", (100, 102, 140, 112))
        read = ocr_word("Basel", (300, 400, 360, 416))
        assert wb._not_drawn([read], page(drawn)) == [read]

    def test_a_word_over_invisible_text_is_kept(self, wb):
        hidden = text_layer_word("Anna", (100, 102, 140, 112), invisible=True)
        read = ocr_word("Anna", (98, 98, 142, 116))
        assert wb._not_drawn([read], page(hidden)) == [read]
