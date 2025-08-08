from __future__ import annotations

from ragqdrant.chunking import AdaptiveHierarchicalChunker


def test_chunking_basic():
    text = ("Hello world.\n\n" * 1000).strip()
    ch = AdaptiveHierarchicalChunker(target_tokens=100, max_tokens=120, overlap_tokens=10)
    chunks = ch.chunk(text)
    assert len(chunks) > 1
    assert all(isinstance(c, str) and len(c) > 0 for c in chunks)