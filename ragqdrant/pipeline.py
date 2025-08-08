from __future__ import annotations

import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from rich.progress import Progress

from .chunking import AdaptiveHierarchicalChunker
from .config import PipelineConfig
from .dedupe import Deduplicator
from .embeddings import EmbeddingService
from .exceptions import ExtractionError, PipelineError
from .logging_config import get_logger
from .metadata import detect_language, optional_spacy_ner, readability_score, simple_keywords
from .metrics import BATCH_LATENCY, CHUNK_COUNT, EMBEDDING_TIME, INGESTED_POINTS, INGEST_ERRORS
from .qdrant_store import QdrantStore
from .schemas import BusinessMetadata, Chunk, ContentMetadata, CoreMetadata, DocumentDescriptor, SemanticMetadata, TechnicalMetadata
from .text_normalization import normalize_text
from .utils import file_stats, stable_hash


log = get_logger(__name__)


class FileLoader:
    SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".html", ".htm"}

    @staticmethod
    def load_text(path: str) -> str:
        ext = Path(path).suffix.lower()
        if ext in {".txt", ".md"}:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        if ext == ".pdf":
            from pypdf import PdfReader  # lazy import

            reader = PdfReader(path)
            text_parts: List[str] = []
            for page in reader.pages:
                try:
                    text_parts.append(page.extract_text() or "")
                except Exception:
                    continue
            return "\n".join(text_parts)
        if ext == ".docx":
            from docx import Document  # type: ignore

            doc = Document(path)
            return "\n".join(p.text for p in doc.paragraphs)
        if ext in {".html", ".htm"}:
            from bs4 import BeautifulSoup  # type: ignore

            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                soup = BeautifulSoup(f, "html.parser")
                for s in soup(["script", "style"]):
                    s.extract()
                return soup.get_text("\n")
        raise ExtractionError(f"Unsupported file extension: {ext}")


class IngestionPipeline:
    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self.chunker = AdaptiveHierarchicalChunker(
            tokenizer_name=cfg.chunking.tokenizer,
            target_tokens=cfg.chunking.target_tokens,
            min_tokens=cfg.chunking.min_tokens,
            max_tokens=cfg.chunking.max_tokens,
            overlap_tokens=cfg.chunking.overlap_tokens,
        )
        self.embedder = EmbeddingService(
            model_name=cfg.embeddings.model_name,
            device=cfg.embeddings.device,
            batch_size=cfg.embeddings.batch_size,
            normalize=cfg.embeddings.normalize_embeddings,
            cache_path=(cfg.embeddings.cache.path if cfg.embeddings.cache.enabled else None),
        )
        self.store = QdrantStore(
            url=cfg.qdrant.url,
            api_key=cfg.qdrant.api_key,
            prefer_grpc=cfg.qdrant.prefer_grpc,
            timeout_sec=cfg.qdrant.timeout_sec,
        )
        self.deduper = Deduplicator(near_duplicate_threshold=cfg.pipeline.near_duplicate_threshold)

    def ensure_collection(self, name: str) -> None:
        self.store.ensure_collection(
            name=name,
            vector_size=self.cfg.collection.vectors.size,
            distance=self.cfg.collection.vectors.distance,
            hnsw_m=self.cfg.collection.vectors.hnsw_config.m,
            hnsw_ef_construct=self.cfg.collection.vectors.hnsw_config.ef_construct,
            optimizers=self.cfg.collection.optimizers_config.model_dump(),
            replication_factor=self.cfg.qdrant.replication_factor,
            write_consistency_factor=self.cfg.qdrant.write_consistency_factor,
            shards_number=self.cfg.collection.shards_number,
        )

    def _make_chunks(self, doc: DocumentDescriptor) -> List[Chunk]:
        raw_chunks = self.chunker.chunk(doc.text)
        CHUNK_COUNT.inc(len(raw_chunks))
        language = detect_language(doc.text) if self.cfg.metadata.enable_language_detection else None
        kw_list = simple_keywords(raw_chunks, top_k=self.cfg.metadata.keywords_top_k)
        entities_list = (
            optional_spacy_ner(raw_chunks, self.cfg.metadata.spacy_model)
            if self.cfg.metadata.enable_spacy_ner
            else [dict() for _ in raw_chunks]
        )

        chunks: List[Chunk] = []
        for idx, text in enumerate(raw_chunks):
            chunk_id = stable_hash(f"{doc.document_id}:{idx}:{text[:50]}")
            core = CoreMetadata(
                document_id=doc.document_id,
                source_path=doc.source_path,
                document_type=doc.document_type,
                created_at=doc.created_at,
                updated_at=doc.updated_at,
                file_size=doc.file_size,
                page_count=doc.page_count,
            )
            content = ContentMetadata(
                language=language,
                content_length=len(text),
                chunk_index=idx,
                chunk_type="paragraph",
            )
            semantic = SemanticMetadata(
                topics=[],
                entities=entities_list[idx],
                keywords=kw_list[idx],
                sentiment_score=None,
                readability_score=readability_score(text),
            )
            technical = TechnicalMetadata(
                embedding_model=self.cfg.embeddings.model_name,
                processing_version=self.cfg.pipeline.processing_version,
                quality_score=None,
            )
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=text,
                    core=core,
                    content=content,
                    semantic=semantic,
                    business=BusinessMetadata(),
                    technical=technical,
                )
            )
        return chunks

    def _embed_chunks(self, chunks: List[Chunk]) -> np.ndarray:
        with EMBEDDING_TIME.time():
            vectors = self.embedder.encode([c.text for c in chunks])
        return vectors

    def _payload(self, chunk: Chunk) -> Dict[str, object]:
        return {
            "text": chunk.text,
            "core": chunk.core.model_dump(),
            "content": chunk.content.model_dump(),
            "semantic": chunk.semantic.model_dump(),
            "business": chunk.business.model_dump(),
            "technical": chunk.technical.model_dump(),
        }

    def ingest_paths(self, paths: Sequence[str], collection: str) -> None:
        self.ensure_collection(collection)
        progress = Progress()
        task_id = progress.add_task("ingesting", total=len(paths))
        progress.start()
        try:
            for path in paths:
                try:
                    size, created_at, updated_at = file_stats(path)
                    text = FileLoader.load_text(path)
                    if not text.strip():
                        progress.advance(task_id)
                        continue
                    doc = DocumentDescriptor(
                        document_id=stable_hash(f"{path}:{updated_at.timestamp()}:{size}"),
                        source_path=os.path.abspath(path),
                        document_type=Path(path).suffix.lower().lstrip("."),
                        created_at=created_at,
                        updated_at=updated_at,
                        file_size=size,
                        page_count=None,
                        text=normalize_text(text),
                    )
                    chunks = self._make_chunks(doc)
                    # Exact dedup within run
                    kept_texts, kept_idx = self.deduper.filter_exact([c.text for c in chunks])
                    if not kept_idx:
                        progress.advance(task_id)
                        continue
                    chunks = [chunks[i] for i in kept_idx]

                    vectors = self._embed_chunks(chunks)
                    # Near-duplicate filtering within document
                    kept_texts2, kept_idx2 = self.deduper.filter_near_duplicates(kept_texts, vectors, self.deduper.threshold)
                    chunks = [chunks[i] for i in kept_idx2]
                    vectors = vectors[kept_idx2]

                    ids = [c.chunk_id for c in chunks]
                    payloads = [self._payload(c) for c in chunks]

                    start = time.perf_counter()
                    self.store.upsert_points(
                        collection=collection,
                        ids=ids,
                        vectors=vectors,
                        payloads=payloads,
                        timeout=self.cfg.pipeline.upsert_timeout_sec,
                    )
                    BATCH_LATENCY.observe(time.perf_counter() - start)
                    INGESTED_POINTS.inc(len(ids))
                except Exception as e:
                    INGEST_ERRORS.inc()
                    log.error("ingest_failed", path=path, error=str(e))
                finally:
                    progress.advance(task_id)
        finally:
            progress.stop()