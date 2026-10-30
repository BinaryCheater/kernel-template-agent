from __future__ import annotations

from pathlib import Path

from kta.core.config import config_from_mapping
from kta.core.context import Context
from kta.core.registry import Registry
from kta.knowledge import import_hw_catalog
from kta.knowledge.pool import names
from kta.store import CodeState, open_store
from kta.targets.gb10 import catalog


def test_catalog_written_once_with_unknown_support(tmp_path: Path) -> None:
    cfg = config_from_mapping({})
    ctx = Context(env="test", config=cfg, data_dir=tmp_path, registry=Registry())
    code = CodeState("0" * 40, False)
    store = open_store(ctx, code)
    features = catalog()
    first = import_hw_catalog(ctx, store, features, code)
    assert first.counts["HwFeature"] == len(features) > 40
    assert all(f.support == {"gb10": "unknown"} for f in features)
    again = import_hw_catalog(ctx, store, features, code)
    assert again.counts == {} and again.written == []
    assert {f.name for f in features} <= names(store, "HwFeature")
