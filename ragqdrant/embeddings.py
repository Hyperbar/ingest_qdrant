from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from typing import Iterable, List, Optional, Sequence

import numpy as np
from fastembed import TextEmbedding

from .exceptions import EmbeddingError
from .utils import stable_hash


class SQLiteEmbeddingCache:
    def __init__(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        with self._conn() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                  key TEXT PRIMARY KEY,
                  vector BLOB NOT NULL
                )
                """
            )

    @contextmanager
    def _conn(self):
        con = sqlite3.connect(self.path)
        try:
            yield con
        finally:
            con.close()

    def get(self, key: str) -> Optional[np.ndarray]:
        with self._conn() as con:
            cur = con.execute("SELECT vector FROM cache WHERE key = ?", (key,))
            row = cur.fetchone()
            if row is None:
                return None
            arr = np.frombuffer(row[0], dtype=np.float32)
            return arr

    def set(self, key: str, vector: np.ndarray) -> None:
        with self._conn() as con:
            con.execute(
                "INSERT OR REPLACE INTO cache(key, vector) VALUES(?, ?)",
                (key, vector.astype(np.float32).tobytes()),
            )
            con.commit()


MODEL_MAP = {
    # Map Sentence-Transformer-like names to fastembed equivalents
    "BAAI/bge-large-en-v1.5": "BAAI/bge-large-en-v1.5",
    "BAAI/bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    "intfloat/e5-large-v2": "intfloat/e5-large-v2",
}


class EmbeddingService:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        batch_size: int = 64,
        normalize: bool = True,
        cache_path: Optional[str] = None,
    ) -> None:
        try:
            self.model_name = MODEL_MAP.get(model_name, model_name)
            self.model = TextEmbedding(self.model_name)
        except Exception as e:
            raise EmbeddingError(f"Failed to load embedding model {model_name}: {e}")
        self.batch_size = batch_size
        self.normalize = normalize
        self.cache = SQLiteEmbeddingCache(cache_path) if cache_path else None

    def _maybe_cache_key(self, text: str) -> Optional[str]:
        if self.cache is None:
            return None
        return stable_hash(text)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        outputs: List[np.ndarray] = []
        start = 0
        while start < len(texts):
            batch = list(texts[start : start + self.batch_size])
            start += self.batch_size
            batch_vecs: List[np.ndarray] = []
            to_compute: List[str] = []
            cache_positions: List[Optional[int]] = []

            if self.cache:
                for i, t in enumerate(batch):
                    key = self._maybe_cache_key(t)
                    vec = self.cache.get(key) if key else None
                    if vec is None:
                        cache_positions.append(None)
                        to_compute.append(t)
                    else:
                        cache_positions.append(i)
                        batch_vecs.append(vec)
            else:
                to_compute = batch

            if to_compute:
                # fastembed returns generator; iterate and stack
                new_vecs_list = list(self.model.embed(to_compute))
                new_vecs = np.array(new_vecs_list, dtype=np.float32)
                if self.normalize:
                    norms = np.linalg.norm(new_vecs, axis=1, keepdims=True) + 1e-9
                    new_vecs = new_vecs / norms
                # Backfill cache
                j = 0
                for idx, pos in enumerate(cache_positions or [None] * len(batch)):
                    if pos is None:
                        vec = new_vecs[j]
                        j += 1
                        batch_vecs.append(vec)
                        if self.cache:
                            key = self._maybe_cache_key(batch[idx])
                            if key:
                                self.cache.set(key, vec)

            outputs.append(np.vstack(batch_vecs))
        return np.vstack(outputs)