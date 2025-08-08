from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class CoreMetadata(BaseModel):
    document_id: str
    source_path: str
    document_type: str
    created_at: datetime
    updated_at: datetime
    file_size: Optional[int] = None
    page_count: Optional[int] = None


class ContentMetadata(BaseModel):
    language: Optional[str] = None
    content_length: int
    chunk_index: int
    chunk_type: str


class SemanticMetadata(BaseModel):
    topics: List[str] = Field(default_factory=list)
    entities: Dict[str, List[str]] = Field(default_factory=dict)
    keywords: List[str] = Field(default_factory=list)
    sentiment_score: Optional[float] = None
    readability_score: Optional[float] = None


class BusinessMetadata(BaseModel):
    department: Optional[str] = None
    confidentiality: Optional[str] = None
    project_tags: List[str] = Field(default_factory=list)
    author: Optional[str] = None


class TechnicalMetadata(BaseModel):
    embedding_model: str
    processing_version: str
    quality_score: Optional[float] = None


class Chunk(BaseModel):
    chunk_id: str
    text: str
    core: CoreMetadata
    content: ContentMetadata
    semantic: SemanticMetadata
    business: BusinessMetadata
    technical: TechnicalMetadata


class DocumentDescriptor(BaseModel):
    document_id: str
    source_path: str
    document_type: str
    created_at: datetime
    updated_at: datetime
    file_size: Optional[int] = None
    page_count: Optional[int] = None
    text: str