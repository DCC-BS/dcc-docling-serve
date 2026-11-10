"""The garbage-collector tuning for docling's PDF pipeline (plugins/pipeline_gc.py)."""

import gc
import importlib.util
import sys
import types
from pathlib import Path

import pytest

PLUGIN = Path(__file__).resolve().parents[2] / "plugins" / "pipeline_gc.py"


@pytest.fixture
def restore_gc():
    threshold = gc.get_threshold()
    finders = list(sys.meta_path)
    yield
    gc.set_threshold(*threshold)
    gc.unfreeze()
    sys.meta_path[:] = finders  # no import hook left behind for later tests


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, PLUGIN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_raises_thresholds_and_freezes_after_pipeline_init(monkeypatch, restore_gc):
    monkeypatch.delenv("DCC_PIPELINE_GC", raising=False)
    module = _load("dcc_pipeline_gc_test")
    assert gc.get_threshold() == (20000, 50, 100)

    built = []
    pipeline = types.ModuleType(module.MODULE)
    pipeline.StandardPdfPipeline = type("StandardPdfPipeline", (), {"__init__": lambda self: built.append(self)})
    module._patch(pipeline)
    module._patch(pipeline)  # a second import does not wrap twice
    gc.unfreeze()
    pipeline.StandardPdfPipeline()

    assert len(built) == 1
    assert gc.get_freeze_count() > 0

    # a later pipeline (another option set) freezes nothing more
    gc.unfreeze()
    pipeline.StandardPdfPipeline()
    assert len(built) == 2
    assert gc.get_freeze_count() == 0


def test_one_import_hook_across_reloads(monkeypatch, restore_gc):
    monkeypatch.delenv("DCC_PIPELINE_GC", raising=False)
    _load("dcc_pipeline_gc_a")
    _load("dcc_pipeline_gc_b")
    assert sum(getattr(f, "dcc_pipeline_gc", False) for f in sys.meta_path) == 1


@pytest.mark.parametrize(("value", "expected"), [("", 20000), ("abc", 20000), ("0", 1), ("5000", 5000)])
def test_threshold_from_env(monkeypatch, restore_gc, value, expected):
    monkeypatch.setenv("DCC_PIPELINE_GC_GEN0", value)
    _load("dcc_pipeline_gc_env")
    assert gc.get_threshold()[0] == expected


def test_off_leaves_the_collector_alone(monkeypatch, restore_gc):
    monkeypatch.setenv("DCC_PIPELINE_GC", "0")
    before = gc.get_threshold()
    finders = list(sys.meta_path)
    _load("dcc_pipeline_gc_off")
    assert gc.get_threshold() == before
    assert sys.meta_path == finders
