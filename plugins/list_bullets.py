"""Recognise Word's Symbol and Wingdings bullets as list markers.

Installed into site-packages as ``dcc_list_bullets.py`` and auto-imported at
interpreter startup through a ``.pth`` file, like ``dcc_word_boxes``.

Word draws list bullets in the Symbol or Wingdings font. A PDF exported from Word
often has no Unicode mapping for these glyphs, so its text layer holds the font's
private-use code point instead of a bullet: U+F0B7 for Symbol's "•", U+F0A7 for
Wingdings' "▪" and so on. docling's ``ListItemMarkerProcessor`` only knows real
bullet characters, so it leaves such a marker in the item's text, and the markdown
export writes ``-  Text`` with an invisible character after the dash.

This adds those private-use bullets to the processor's bullet patterns, so the
marker is split off like any other bullet, and records it as "•" so the private-use
character does not leave the server in ``ListItem.marker`` either.

The patch is applied lazily, when docling imports the processor module, so
interpreters that never load it pay nothing.

Env vars:
    DCC_LIST_BULLETS=0   leave docling's list markers as upstream has them
"""

import functools
import importlib.abc
import importlib.machinery
import logging
import os
import re
import sys

_logger = logging.getLogger(__name__)

MODULE = "docling.models.postprocessing.list_marker_processor"

# Private-use code points of bullet glyphs in the Symbol (F0B7) and Wingdings fonts
PRIVATE_USE_BULLETS = {
    "": "•",  # Symbol: bullet
    "": "▪",  # Wingdings: small square
    "": "■",  # Wingdings: black square
    "": "❖",  # Wingdings: diamond minus white X
    "": "➢",  # Wingdings: arrowhead
    "": "➔",  # Wingdings: arrow
    "": "✓",  # Wingdings: check mark
    "": "◦",  # Wingdings: white bullet
}
_PATTERN = "[" + "".join(PRIVATE_USE_BULLETS) + "]"


def _enabled() -> bool:
    return os.environ.get("DCC_LIST_BULLETS", "1").strip().lower() not in {"0", "false", "no", "off"}


def _patch(module) -> None:
    processor = module.ListItemMarkerProcessor
    if getattr(processor, "_dcc_list_bullets", False):
        return
    original_init = processor.__init__
    original_process = processor.process_list_item

    @functools.wraps(original_init)
    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # The compiled lists are built from _bullet_patterns in __init__; extend all three alike
        self._bullet_patterns.append(_PATTERN)
        self._compiled_bullet_patterns.append(re.compile(f"^{_PATTERN}$"))
        self._compiled_bullet_item_patterns.append(re.compile(f"^({_PATTERN})" + r"\s(.+)", re.DOTALL))

    @functools.wraps(original_process)
    def process_list_item(self, item):
        item = original_process(self, item)
        marker = getattr(item, "marker", None)
        if marker in PRIVATE_USE_BULLETS:
            item.marker = PRIVATE_USE_BULLETS[marker]
        return item

    processor.__init__ = __init__
    processor.process_list_item = process_list_item
    processor._dcc_list_bullets = True


class _PatchOnImport(importlib.abc.MetaPathFinder):
    """Apply the patch right after the processor module is executed."""

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
                _logger.exception("Could not add private-use bullets to %s", fullname)

        spec.loader.exec_module = exec_module
        return spec


def _install() -> None:
    if not _enabled():
        return
    if MODULE in sys.modules:
        try:
            _patch(sys.modules[MODULE])
        except Exception:
            _logger.exception("Could not add private-use bullets to %s", MODULE)
    if not any(isinstance(finder, _PatchOnImport) for finder in sys.meta_path):
        sys.meta_path.insert(0, _PatchOnImport())


_install()
