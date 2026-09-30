"""Let the convert API hand back the words of each page.

Installed into site-packages as ``dcc_word_boxes.py`` and auto-imported at
interpreter startup through a ``.pth`` file, like ``dcc_ocr_ignore_shapes``.

docling knows where every word of a PDF stands: the parser measures a box for
each one, and OCR adds what only the pixels show. None of it leaves the server.
``DoclingDocument`` carries regions and their text, not words, so whoever needs
a word's place has to parse the file a second time on their own side, and run
their own OCR for the pages that have no text layer.

This adds one request option, ``include_word_boxes``, and one field beside the
other contents of the answer, ``word_boxes``: per page number, docling's own
``SegmentedPdfPage`` with ``word_cells`` and ``textline_cells``, each cell with
its text, the rectangle it was read in and whether OCR read it. The characters
and the page picture are dropped: they are large, and a word's text and box is
all anyone asked for.

The pipeline throws these cells away after assembly unless it is told to keep
them, so the option also turns on ``generate_parsed_pages``.

The text layer measures its own words; OCR only reads lines. With RapidOCR the
option also asks for a box per word, reads the whole page instead of the parts
the layout found, and adds the words it read where the text layer draws none.
Other engines give lines only, so their pages carry the text layer's words.

Everything is off unless a request asks for it: the option defaults to false, no
existing field changes, and a request that does not mention it is answered
exactly as before.

The patch is applied lazily, when docling and docling-jobkit import the modules
it touches, so interpreters that never load them pay nothing.

Env vars:
    DCC_WORD_BOXES=0   leave the API as upstream has it
"""

import functools
import importlib.abc
import importlib.machinery
import logging
import os
import sys
from typing import Any, Optional, get_args

_logger = logging.getLogger("dcc_word_boxes")

#: The request option and the answer's field.
OPTION = "include_word_boxes"
FIELD = "word_boxes"


def _enabled() -> bool:
    return os.environ.get("DCC_WORD_BOXES", "1").strip().lower() not in ("0", "false", "no", "off")


def _add_field(model, name: str, annotation, default, description: str, module) -> None:
    """Give a pydantic model one more field, and its holders the news.

    A model that holds another one keeps a copy of that one's schema, taken when
    it was first built. Adding the field to the held model alone would let it be
    set and then dropped again the moment it is answered inside its holder, so
    every model that reaches this one is built afresh too — those, and no others:
    building a model is not free, and most of a module has nothing to do with us.
    """
    from pydantic.fields import FieldInfo

    if name in model.model_fields:
        return
    model.model_fields[name] = FieldInfo(annotation=annotation, default=default, description=description)
    model.model_rebuild(force=True)
    for holder in _holders_of(model, module):
        try:
            holder.model_rebuild(force=True)
        except Exception:  # noqa: BLE001 - one model that cannot be built again stops nothing
            _logger.debug("Could not rebuild %s after adding %s", holder.__name__, name)


def _holders_of(model, module) -> list:
    """The module's models that reach ``model`` through their fields, held first.

    A holder of a holder counts: the field travels up every step of the way to
    the answer, and each step keeps its own copy of the schema below it.
    """
    from pydantic import BaseModel

    def classes(annotation) -> set:
        """Every class an annotation is made of, through lists, dicts and optionals."""
        parts = get_args(annotation)
        if parts:
            return set().union(*(classes(part) for part in parts))
        return {annotation} if isinstance(annotation, type) else set()

    def holds(candidate, known: set) -> bool:
        return any(classes(field.annotation) & known for field in candidate.model_fields.values())

    # dict.fromkeys: a module may bind the same model under two names.
    rest = list(
        dict.fromkeys(
            value
            for value in vars(module).values()
            if isinstance(value, type) and issubclass(value, BaseModel) and value.__module__ == module.__name__
        )
    )
    holders: list = []
    reached = {model}
    while True:
        more = [candidate for candidate in rest if holds(candidate, reached)]
        if not more:
            return holders
        holders += more
        reached.update(more)
        rest = [candidate for candidate in rest if candidate not in reached]


def _patch_options(module) -> None:
    """``include_word_boxes`` on the convert request."""
    options = getattr(module, "ConvertDocumentsOptions", None)
    if options is None:
        _logger.warning("docling has no ConvertDocumentsOptions; word_boxes option not added")
        return
    _add_field(
        options,
        OPTION,
        bool,
        False,
        "If enabled, the answer carries the words of every page: their text and the rectangle each "
        "was read in, from the text layer and from OCR. Boolean. Optional, defaults to false.",
        module,
    )


def _patch_responses(module) -> None:
    """``word_boxes`` beside the other contents of the answer."""
    from docling_core.types.doc.page import SegmentedPdfPage

    response = getattr(module, "ExportDocumentResponse", None)
    if response is None:
        _logger.warning("docling has no ExportDocumentResponse; word_boxes field not added")
        return
    _add_field(
        response,
        FIELD,
        Optional[dict[int, SegmentedPdfPage]],
        None,
        "Words and text lines of each page, when include_word_boxes asked for them.",
        module,
    )


def _patch_exportable(module) -> None:
    """Carry the pages off the conversion result, on their way to the answer."""
    from docling_core.types.doc.page import SegmentedPdfPage

    exportable = getattr(module, "ExportableDocument", None)
    original = getattr(exportable, "from_conversion_result", None)
    if original is None:
        _logger.warning("docling-jobkit has no ExportableDocument.from_conversion_result; pages not carried")
        return
    if getattr(original, "_dcc_carries_word_boxes", False):
        return
    _add_field(
        exportable,
        FIELD,
        Optional[dict[int, SegmentedPdfPage]],
        None,
        "Words and text lines of each page",
        module,
    )

    @functools.wraps(original.__func__)
    def from_conversion_result(cls, conversion_result, **kwargs):
        document = original(conversion_result, **kwargs)
        pages = {
            page.page_no: page.parsed_page.model_copy(update={"char_cells": [], "image": None})
            for page in conversion_result.pages
            if getattr(page, "parsed_page", None) is not None
        }
        if pages and document.document is not None:
            setattr(document, FIELD, pages)
        return document

    from_conversion_result._dcc_carries_word_boxes = True
    exportable.from_conversion_result = classmethod(from_conversion_result)


#: RapidOCR's own setting for a box per word. Set only on requests that ask for
#: word boxes, it also tells the model below to read the whole page and keep
#: the words.
RAPIDOCR_WORD_BOXES = "Global.return_word_box"


def _ocr_for_word_boxes(ocr_options):
    """The request's OCR options, told to place every word it reads.

    Only RapidOCR is told: no other engine docling ships says where a single
    word stands. docling-serve keeps one converter per set of options, so
    requests without word boxes keep theirs and read as before.
    """
    if getattr(ocr_options, "kind", None) != "rapidocr":
        return ocr_options
    params = {**ocr_options.rapidocr_params, RAPIDOCR_WORD_BOXES: True}
    return ocr_options.model_copy(update={"rapidocr_params": params})


def _reads_words(model) -> bool:
    return bool((getattr(model.options, "rapidocr_params", None) or {}).get(RAPIDOCR_WORD_BOXES))


def _ocr_words(word_results, scale: float) -> list:
    """The words RapidOCR read on a whole page, in the page's points.

    RapidOCR answers per line, each word as its text, how sure it is and the
    four corners it was read in. A page it read nothing on comes back as one
    bare word, ``('', 1.0, None)``, not as a line of them; a word it could not
    place comes without corners. Neither has a place to give. The caller
    numbers the words, after those of the text layer.
    """
    from docling_core.types.doc import CoordOrigin
    from docling_core.types.doc.page import BoundingRectangle, TextCell

    words = []
    for line in word_results or ():
        if line and isinstance(line[0], str):
            line = (line,)
        for word in line:
            if not isinstance(word, (tuple, list)) or len(word) != 3:
                continue
            text, confidence, corners = word
            if corners is None or not str(text).strip():
                continue
            (x0, y0), (x1, y1), (x2, y2), (x3, y3) = ((x / scale, y / scale) for x, y in corners)
            words.append(
                TextCell(
                    index=0,
                    text=text,
                    orig=text,
                    confidence=confidence,
                    from_ocr=True,
                    rect=BoundingRectangle(
                        r_x0=x0,
                        r_y0=y0,
                        r_x1=x1,
                        r_y1=y1,
                        r_x2=x2,
                        r_y2=y2,
                        r_x3=x3,
                        r_y3=y3,
                        coord_origin=CoordOrigin.TOPLEFT,
                    ),
                )
            )
    return words


def _not_drawn(words: list, page) -> list:
    """The words no word of the text layer is drawn on.

    Read whole, a page is read where its text layer draws the words too, and
    those are exact already. OCR reads a word in a box as tall as its line and
    the text layer in one that hugs the ink, so they are the same word where
    the OCR word's centre lies in a drawn one. Invisible text, the layer a
    scanner lays under its picture, is not drawn: the words read in the
    picture stay, as the only ones that show where the picture has them.
    """
    from docling_core.types.doc.page import PdfCellRenderingMode

    height = page.size.height
    drawn = [
        cell.rect.to_top_left_origin(height).to_bounding_box()
        for cell in page.parsed_page.word_cells
        if not cell.from_ocr and getattr(cell, "rendering_mode", None) != PdfCellRenderingMode.INVISIBLE
    ]

    def on_drawn(word) -> bool:
        box = word.rect.to_bounding_box()
        x, y = (box.l + box.r) / 2, (box.t + box.b) / 2
        return any(other.l <= x <= other.r and other.t <= y <= other.b for other in drawn)

    return [word for word in words if not on_drawn(word)]


class _Recording:
    """RapidOCR's reader, keeping the words of what it read for the page."""

    def __init__(self, reader) -> None:
        self.reader = reader
        self.word_results: list = []

    def __call__(self, *args: Any, **kwargs: Any):
        result = self.reader(*args, **kwargs)
        self.word_results.append(getattr(result, "word_results", None))
        return result


def _patch_rapidocr(module) -> None:
    """Let RapidOCR read the whole page and give each word its place."""
    model = getattr(module, "RapidOcrModel", None)
    original_rects = getattr(model, "get_ocr_rects", None)
    original_call = model.__call__ if model is not None else None
    if original_rects is None or original_call is None:
        _logger.warning("docling has no RapidOcrModel to patch; RapidOCR reads no word boxes")
        return
    if getattr(original_call, "_dcc_reads_words", False):
        return

    @functools.wraps(original_rects)
    def get_ocr_rects(self, page):
        """The whole page, not the parts the layout found: a line of text cut
        by a layout box is read only as far as the cut."""
        if _reads_words(self) and page.size is not None:
            from docling_core.types.doc import BoundingBox, CoordOrigin

            size = page.size
            return [BoundingBox(l=0, t=0, r=size.width, b=size.height, coord_origin=CoordOrigin.TOPLEFT)]
        return original_rects(self, page)

    @functools.wraps(original_call)
    def call(self, conv_res, page_batch):
        if not _reads_words(self) or getattr(self, "reader", None) is None:
            yield from original_call(self, conv_res, page_batch)
            return
        if not isinstance(self.reader, _Recording):
            self.reader = _Recording(self.reader)
        for page in page_batch:
            self.reader.word_results.clear()
            for done in original_call(self, conv_res, [page]):
                if done.parsed_page is not None and done.size is not None:
                    cells = done.parsed_page.word_cells
                    read = [word for results in self.reader.word_results for word in _ocr_words(results, self.scale)]
                    words = _not_drawn(read, done)
                    for index, word in enumerate(words):
                        word.index = len(cells) + index
                    done.parsed_page.word_cells = [*cells, *words]
                    done.parsed_page.has_words = bool(done.parsed_page.word_cells)
                yield done

    call._dcc_reads_words = True
    model.get_ocr_rects = get_ocr_rects
    model.__call__ = call


def _patch_manager(module) -> None:
    """Keep the parsed cells when a request asked for the words."""
    manager = getattr(module, "DoclingConverterManager", None)
    original = getattr(manager, "_parse_standard_pdf_opts", None)
    if original is None:
        _logger.warning("docling-jobkit has no _parse_standard_pdf_opts; word boxes will be empty")
        return
    if getattr(original, "_dcc_keeps_word_boxes", False):
        return

    @functools.wraps(original)
    def parse_standard_pdf_opts(self, request, artifacts_path):
        options = original(self, request, artifacts_path)
        if getattr(request, OPTION, False):
            options.generate_parsed_pages = True
            options.ocr_options = _ocr_for_word_boxes(options.ocr_options)
        return options

    parse_standard_pdf_opts._dcc_keeps_word_boxes = True
    manager._parse_standard_pdf_opts = parse_standard_pdf_opts


def _patch_results(module) -> None:
    """Put the pages in the answer."""
    original = getattr(module, "_export_document_as_content", None)
    if original is None:
        _logger.warning("docling-jobkit has no _export_document_as_content; word boxes not answered")
        return
    if getattr(original, "_dcc_answers_word_boxes", False):
        return

    @functools.wraps(original)
    def export_document_as_content(exportable_document, *args: Any, **kwargs: Any):
        document = original(exportable_document, *args, **kwargs)
        pages = getattr(exportable_document, FIELD, None)
        if pages:
            setattr(document, FIELD, pages)
        return document

    export_document_as_content._dcc_answers_word_boxes = True
    module._export_document_as_content = export_document_as_content


#: Every module to hook, and what to do once it is there.
PATCHES = {
    "docling.datamodel.service.options": _patch_options,
    "docling.datamodel.service.responses": _patch_responses,
    "docling_jobkit.datamodel.exportable_document": _patch_exportable,
    "docling_jobkit.convert.manager": _patch_manager,
    "docling_jobkit.convert.results": _patch_results,
    "docling.models.stages.ocr.rapid_ocr_model": _patch_rapidocr,
}


class _PatchOnImport(importlib.abc.MetaPathFinder):
    """Apply each patch right after its module is executed."""

    def find_spec(self, fullname, path, target=None):
        patch = PATCHES.get(fullname)
        if patch is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        loader_exec = spec.loader.exec_module

        def exec_module(module):
            loader_exec(module)
            try:
                patch(module)
            except Exception:  # never break docling because of this patch
                _logger.exception("Could not add %s for %s", FIELD, fullname)

        spec.loader.exec_module = exec_module
        return spec


def _install() -> None:
    if not _enabled():
        return
    for name, patch in PATCHES.items():
        if name in sys.modules:
            try:
                patch(sys.modules[name])
            except Exception:
                _logger.exception("Could not add %s for %s", FIELD, name)
    if not any(isinstance(finder, _PatchOnImport) for finder in sys.meta_path):
        sys.meta_path.insert(0, _PatchOnImport())


_install()
