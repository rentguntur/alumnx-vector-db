from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class RetrievalStrategy(BaseModel):
    algorithm: str = "knn"
    distance_metric: str = "cosine"


class StrategyResult(BaseModel):
    strategy_name: str
    chunk_count: int
    embedding_model: str
    vector_size: int
    overwritten: bool


class IngestResponse(BaseModel):
    kb_name: str
    source_filename: str
    strategies_processed: list[StrategyResult]
    ingested_at: str


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    kb_name: Optional[str] = None
    retrieval_strategy: Optional[RetrievalStrategy] = None
    k: Optional[int] = Field(default=None, ge=1)
    embedding_model: Optional[str] = None
    excludevectors: bool = False


class ChunkResult(BaseModel):
    chunk_id: str
    similarity_score: float
    chunk_text: str
    embedding_vector: list[float]
    source_filename: str
    chunk_index: int
    page_number: Optional[int]
    created_at: str


class StrategyGroupResult(BaseModel):
    chunking_strategy: str
    embedding_model: str
    chunks: list[ChunkResult]


class KBResult(BaseModel):
    kb_name: str
    strategy_results: list[StrategyGroupResult]


class RetrieveResponse(BaseModel):
    query: str
    retrieval_strategy_used: RetrievalStrategy
    k_used: int
    results: list[KBResult]


class ErrorDetail(BaseModel):
    source_filename: Optional[str] = None
    chunking_strategy: Optional[str] = None
    embedding_model: Optional[str] = None
    kb_name: Optional[str] = None


class ErrorResponse(BaseModel):
    error: str
    message: str
    detail: Optional[ErrorDetail] = None


class DocumentResponse(BaseModel):
    id: str = Field(..., description="Unique UUID for this document record.")
    file_hash: str = Field(..., description="SHA-256 hash of the physical file to prevent exact duplicates.")
    original_filename: str = Field(..., description="The original filename of the document as uploaded.")
    title: str = Field(..., description="Human readable title for the document. Defaults to original_filename if empty.")
    description: Optional[str] = Field(None, description="Optional notes or context about the document.")
    kb_name: Optional[str] = Field(None, description="Optional Knowledge Base name this document belongs to, useful for grouping files before vectorization.")
    status: str = Field(..., description="Current status of the document lifecycle, e.g., 'UPLOADED'.")
    file_size_bytes: int = Field(..., description="Physical file size in bytes.")
    created_at: str = Field(..., description="ISO 8601 timestamp of when the document was uploaded.")
