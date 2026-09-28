"""Stop vector shapes from sending programmatic PDF text to OCR.

Installed into site-packages as ``dcc_ocr_ignore_shapes.py`` and auto-imported at
interpreter startup through a ``.pth`` file, like ``dcc_mute_health_logs``.

docling's default OCR mode (pdf_aware_layout_regions) OCRs every layout region that
overlaps a bitmap or a vector shape, even when the region already carries PDF text.
Table rules, underlines and background fills are shapes, so on many born-digital
documents almost every region is OCR'd. docling keeps the PDF text wherever both exist,
so that OCR is thrown away. With this patch shapes no longer count as "non-text
content" for the OCR input selection; bitmaps and regions without PDF text are still
OCR'd exactly as before.

The patch is applied lazily, when docling imports ``docling.models.base_ocr_model``,
so interpreters that never load docling pay nothing.

Env vars:
    DCC_OCR_IGNORE_SHAPES=0   keep docling's behaviour (shapes trigger OCR)
"""

import functools
import importlib.abc
import importlib.machinery
import logging
import os
import sys

_TARGET_MODULE = "docling.models.base_ocr_model"
_logger = logging.getLogger("dcc_ocr_ignore_shapes")


def _enabled() -> bool:
    return os.environ.get("DCC_OCR_IGNORE_SHAPES", "1").strip().lower() not in ("0", "false", "no", "off")


def _patch(module) -> None:
    base = getattr(module, "BaseOcrModel", None)
    original = getattr(base, "_find_pdf_aware_layout_ocr_rects", None)
    if original is None or getattr(original, "_dcc_ignores_shapes", False):
        if original is None:
            _logger.warning("docling has no BaseOcrModel._find_pdf_aware_layout_ocr_rects; shape patch not applied")
        return

    @functools.wraps(original)
    def find_ocr_rects_ignoring_shapes(self, page):
        backend = getattr(page, "_backend", None)
        if backend is None:
            return original(self, page)

        backend_has_content_in = backend.has_content_in

        def has_content_in(*, bbox, chars=True, shapes=True, bitmaps=True):
            if not chars and not bitmaps:
                # Pure shape query: report "nothing" rather than "not supported".
                result = backend_has_content_in(bbox=bbox, chars=False, shapes=False, bitmaps=True)
                return None if result is None else False
            return backend_has_content_in(bbox=bbox, chars=chars, shapes=False, bitmaps=bitmaps)

        # Instance attributes shadow the class methods for this one call only.
        backend.has_content_in = has_content_in
        backend.get_connected_shape_bounding_boxes = lambda: []
        try:
            return original(self, page)
        finally:
            del backend.has_content_in
            del backend.get_connected_shape_bounding_boxes

    find_ocr_rects_ignoring_shapes._dcc_ignores_shapes = True
    base._find_pdf_aware_layout_ocr_rects = find_ocr_rects_ignoring_shapes
    _logger.info("Vector shapes no longer trigger OCR (DCC_OCR_IGNORE_SHAPES=0 to disable)")


class _PatchOnImport(importlib.abc.MetaPathFinder):
    """Apply the patch right after docling's OCR base module is executed."""

    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET_MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        loader_exec = spec.loader.exec_module

        def exec_module(module):
            loader_exec(module)
            try:
                _patch(module)
            except Exception:  # never break docling because of this patch
                _logger.exception("Could not apply the OCR shape patch")

        spec.loader.exec_module = exec_module
        return spec


def _install() -> None:
    if not _enabled():
        return
    if _TARGET_MODULE in sys.modules:
        _patch(sys.modules[_TARGET_MODULE])
    elif not any(isinstance(finder, _PatchOnImport) for finder in sys.meta_path):
        sys.meta_path.insert(0, _PatchOnImport())


_install()
