from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np

from .utils import stable_hash


class Deduplicator:
    def __init__(self, near_duplicate_threshold: float = 0.98) -> None:
        self.seen_hashes: set[str] = set()
        self.threshold = near_duplicate_threshold

    def filter_exact(self, texts: List[str]) -> Tuple[List[str], List[int]]:
        kept: List[str] = []
        kept_idx: List[int] = []
        for i, t in enumerate(texts):
            h = stable_hash(t)
            if h in self.seen_hashes:
                continue
            self.seen_hashes.add(h)
            kept.append(t)
            kept_idx.append(i)
        return kept, kept_idx

    @staticmethod
    def filter_near_duplicates(
        texts: List[str], vectors: np.ndarray, threshold: float
    ) -> Tuple[List[str], List[int]]:
        if len(texts) <= 1:
            return texts, list(range(len(texts)))
        kept_texts: List[str] = []
        kept_idx: List[int] = []
        used: set[int] = set()
        # Greedy: keep a representative if similarity to previous kept is below threshold
        for i in range(len(texts)):
            if i in used:
                continue
            vi = vectors[i]
            keep = True
            for j in kept_idx:
                vj = vectors[j]
                sim = float(np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-9))
                if sim >= threshold:
                    keep = False
                    break
            if keep:
                kept_texts.append(texts[i])
                kept_idx.append(i)
                used.add(i)
        return kept_texts, kept_idx