from __future__ import annotations

import re
from typing import Iterable

_whitespace_re = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u00A0", " ")
    text = _whitespace_re.sub(" ", text)
    return text.strip()


def strip_control_characters(text: str) -> str:
    return "".join(ch for ch in text if ch.isprintable() or ch in {"\n", "\t", " "})


def normalize_text(text: str) -> str:
    return normalize_whitespace(strip_control_characters(text))