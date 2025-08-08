from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .text_normalization import normalize_text


@dataclass
class ChunkSpec:
    start: int
    end: int
    text: str


class AdaptiveHierarchicalChunker:
    def __init__(
        self,
        tokenizer_name: str = "cl100k_base",
        target_tokens: int = 350,
        min_tokens: int = 120,
        max_tokens: int = 512,
        overlap_tokens: int = 40,
    ) -> None:
        # Approximate tokens by words (OpenAI ~ 0.75 words/token). We'll use words to avoid heavy deps.
        self.target_words = int(target_tokens * 0.75)
        self.min_words = int(min_tokens * 0.75)
        self.max_words = int(max_tokens * 0.75)
        self.overlap_words = int(overlap_tokens * 0.75)

    def _split_by_paragraphs(self, text: str) -> List[str]:
        parts = [p.strip() for p in text.split("\n\n")]
        return [p for p in parts if p]

    def _split_words(self, text: str) -> List[str]:
        return text.split()

    def _merge_segments(self, segments: List[str]) -> List[ChunkSpec]:
        chunks: List[ChunkSpec] = []
        buffer_words: List[str] = []
        cursor = 0

        def flush_buffer(end_cursor: int) -> None:
            nonlocal buffer_words, cursor
            if not buffer_words:
                return
            txt = " ".join(buffer_words)
            chunks.append(ChunkSpec(start=cursor, end=end_cursor, text=txt))
            cursor = end_cursor
            buffer_words = []

        for seg in segments:
            seg = normalize_text(seg)
            words = self._split_words(seg)
            if len(words) > self.max_words:
                # Hard split long paragraphs
                start = 0
                while start < len(words):
                    end = min(start + self.max_words, len(words))
                    piece = " ".join(words[start:end])
                    chunks.append(ChunkSpec(start=cursor, end=cursor + len(piece), text=piece))
                    start = max(end - self.overlap_words, end)
                    cursor += len(piece)
                continue
            if len(buffer_words) + len(words) <= self.target_words:
                buffer_words.extend(words)
            else:
                flush_buffer(cursor)
                buffer_words = words
        flush_buffer(cursor)
        return chunks

    def chunk(self, text: str) -> List[str]:
        text = normalize_text(text)
        segments = self._split_by_paragraphs(text)
        coarse = self._merge_segments(segments)
        return [c.text for c in coarse]