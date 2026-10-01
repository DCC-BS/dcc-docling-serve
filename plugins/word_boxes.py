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
them, so the option also turns on ``generate_parsed_pages``. With PP-OCRv6 it
also turns on ``whole_page`` and ``return_word_box``, so a scanned word is read
whole and gets a box of its own; requests without the option keep the engine's
defaults and its speed.

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


#: OCR options that make an engine read single words whole (docling-pp-ocrv6).
#: Off by default in the engine; set only for requests that ask for word boxes.
WORD_BOX_OCR_OPTIONS = ("whole_page", "return_word_box")


def _ocr_for_word_boxes(ocr_options):
    """The request's OCR options, told to read every word whole and place it.

    Only engines that know these options get them, so other engines and a
    server without the plugin are left as they are. docling-serve keeps one
    converter per set of options, so requests without word boxes keep theirs.
    """
    fields = getattr(type(ocr_options), "model_fields", {})
    update = {name: True for name in WORD_BOX_OCR_OPTIONS if name in fields}
    return ocr_options.model_copy(update=update) if update else ocr_options


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
