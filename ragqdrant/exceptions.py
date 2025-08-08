from typing import Optional


class PipelineError(Exception):
    """Base exception for pipeline-related errors."""


class ConfigurationError(PipelineError):
    """Raised when configuration is missing or invalid."""


class ExtractionError(PipelineError):
    """Raised when document extraction fails."""


class EmbeddingError(PipelineError):
    """Raised when embedding computation fails."""


class QdrantError(PipelineError):
    """Raised when Qdrant operations fail."""


class DeduplicationError(PipelineError):
    """Raised when deduplication logic fails."""


class ValidationError(PipelineError):
    """Raised when data validation fails prior to ingestion."""