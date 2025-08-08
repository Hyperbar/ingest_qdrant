from __future__ import annotations

import hashlib
import os
from datetime import datetime
from typing import Tuple


def file_stats(path: str) -> Tuple[int, datetime, datetime]:
    st = os.stat(path)
    created_at = datetime.fromtimestamp(getattr(st, "st_ctime", st.st_mtime))
    updated_at = datetime.fromtimestamp(st.st_mtime)
    return st.st_size, created_at, updated_at


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()