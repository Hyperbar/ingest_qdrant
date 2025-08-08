from __future__ import annotations

from ragqdrant.metadata import detect_language, readability_score, simple_keywords


def test_language_detection():
    assert detect_language("This is a sentence in English.") in {"en", None}


def test_readability_score():
    s = readability_score("This is a short sentence. Another short sentence.")
    assert s >= 0.0


def test_simple_keywords():
    kws = simple_keywords(["apple banana apple", "car bus car"])
    assert len(kws) == 2