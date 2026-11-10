"""Fewer stop-the-world garbage collections in docling's threaded PDF pipeline.

Installed into site-packages as ``dcc_pipeline_gc.py`` and auto-imported at interpreter
startup through a ``.pth`` file, like ``dcc_list_bullets``.

Python's cyclic garbage collector runs a collection every 700 net allocations
(generation 0) and walks the whole heap, models and every page in flight included,
every 100 of those. docling's pages allocate millions of small objects (text cells,
clusters), so a 600-page PDF triggered about 9,000 collections that held the GIL for
8 s while all pipeline stages stood still. This raises the generation-0 threshold to
20,000 (the older generations' to 50 and 100) and freezes everything alive once a
pipeline has loaded its models (``gc.freeze``, once per process), so later full
collections do not walk the model objects again. What a conversion returns does not
change, only when the memory of dropped cycles is reclaimed.

The patch is applied lazily, when docling imports its standard PDF pipeline.

Env vars:
    DCC_PIPELINE_GC=0          leave the garbage collector as Python has it
    DCC_PIPELINE_GC_GEN0=20000 generation-0 threshold (Python's default is 700)
"""

import functools
import gc
import importlib.abc
import importlib.machinery
import logging
import os
import sys
import threading

_logger = logging.getLogger("dcc_pipeline_gc")
MODULE = "docling.pipeline.standard_pdf_pipeline"


def _enabled() -> bool:
    return os.environ.get("DCC_PIPELINE_GC", "1").strip().lower() not in ("0", "false", "no", "off")


_frozen = False
_frozen_lock = threading.Lock()


def _freeze_once() -> None:
    """After the first pipeline has loaded its models: collect once, then keep them out of later collections.

    Only the first time: a later pipeline (docling-serve builds one per new option set) would also
    freeze the pages and results of conversions running at that moment, and anything frozen is
    never collected again, so a dropped pipeline's models would stay in memory.
    """
    global _frozen
    with _frozen_lock:
        if _frozen:
            return
        _frozen = True
    gc.collect()
    gc.freeze()


def _gen0() -> int:
    try:
        return max(1, int(os.environ.get("DCC_PIPELINE_GC_GEN0", "").strip()))
    except ValueError:
        return 20000


def _patch(module) -> None:
    cls = module.StandardPdfPipeline
    if getattr(cls, "_dcc_pipeline_gc", False):
        return
    init = cls.__init__

    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        init(self, *args, **kwargs)
        _freeze_once()

    cls.__init__ = __init__
    cls._dcc_pipeline_gc = True


class _PatchOnImport(importlib.abc.MetaPathFinder):
    """Apply the patch right after the pipeline module is executed."""

    dcc_pipeline_gc = True  # recognised across reloads of this module

    def find_spec(self, fullname, path, target=None):
        if fullname != MODULE:
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
                _logger.exception("Could not tune the garbage collector for %s", fullname)

        spec.loader.exec_module = exec_module
        return spec


def _install() -> None:
    if not _enabled():
        return
    # older generations collected after 50 and 100 younger collections (Python: 10 and 10)
    gc.set_threshold(_gen0(), 50, 100)
    if MODULE in sys.modules:
        try:
            _patch(sys.modules[MODULE])
        except Exception:
            _logger.exception("Could not tune the garbage collector for %s", MODULE)
    if not any(
        type(finder).__name__ == "_PatchOnImport" and getattr(finder, "dcc_pipeline_gc", False)
        for finder in sys.meta_path
    ):
        sys.meta_path.insert(0, _PatchOnImport())


_install()
