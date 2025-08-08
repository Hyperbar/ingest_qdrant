from __future__ import annotations

from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError as PydanticValidationError

from .exceptions import ConfigurationError


class HNSWConfig(BaseModel):
    m: int = 32
    ef_construct: int = 256


class VectorsConfig(BaseModel):
    size: int
    distance: str = "Cosine"
    hnsw_config: HNSWConfig = Field(default_factory=HNSWConfig)


class OptimizersConfig(BaseModel):
    deleted_threshold: float = 0.2
    vacuum_min_vector_number: int = 1000
    default_segment_number: int = 2


class CollectionConfig(BaseModel):
    name: str
    vectors: VectorsConfig
    optimizers_config: OptimizersConfig = Field(default_factory=OptimizersConfig)
    shards_number: int = 2


class QdrantConfig(BaseModel):
    url: str = "http://localhost:6333"
    api_key: Optional[str] = None
    prefer_grpc: bool = True
    timeout_sec: int = 30
    replication_factor: int = 2
    write_consistency_factor: int = 1


class EmbeddingCacheConfig(BaseModel):
    enabled: bool = True
    path: str = ".cache/embeddings.sqlite"


class EmbeddingConfig(BaseModel):
    model_name: str = "BAAI/bge-large-en-v1.5"
    batch_size: int = 64
    normalize_embeddings: bool = True
    device: str = "auto"
    cache: EmbeddingCacheConfig = Field(default_factory=EmbeddingCacheConfig)


class ChunkingConfig(BaseModel):
    strategy: str = "adaptive_hierarchical"
    target_tokens: int = 350
    min_tokens: int = 120
    max_tokens: int = 512
    overlap_tokens: int = 40
    tokenizer: str = "cl100k_base"


class MetadataConfig(BaseModel):
    enable_language_detection: bool = True
    enable_spacy_ner: bool = False
    spacy_model: str = "en_core_web_sm"
    keywords_top_k: int = 10


class PipelineRuntimeConfig(BaseModel):
    processing_version: str = "v1"
    batch_size_points: int = 256
    max_concurrency: int = 4
    near_duplicate_threshold: float = 0.98
    upsert_timeout_sec: int = 60


class MetricsConfig(BaseModel):
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8008


class PipelineConfig(BaseModel):
    qdrant: QdrantConfig
    collection: CollectionConfig
    embeddings: EmbeddingConfig
    chunking: ChunkingConfig
    metadata: MetadataConfig
    pipeline: PipelineRuntimeConfig
    metrics: MetricsConfig


def load_config(path: str) -> PipelineConfig:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return PipelineConfig(**data)
    except FileNotFoundError as e:
        raise ConfigurationError(f"Config file not found: {path}") from e
    except PydanticValidationError as e:
        raise ConfigurationError(f"Invalid configuration: {e}") from e
    except Exception as e:
        raise ConfigurationError(f"Failed to load configuration: {e}") from e