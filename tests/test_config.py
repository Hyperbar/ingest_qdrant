from __future__ import annotations

from ragqdrant.config import load_config


def test_load_config():
    cfg = load_config("config/pipeline.yaml")
    assert cfg.collection.vectors.size > 0
    assert cfg.qdrant.url.startswith("http")