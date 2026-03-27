from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Ingest models
# ---------------------------------------------------------------------------

class SectionResult(BaseModel):
    section_name: str
    chunk_id: str


class IngestResponse(BaseModel):
    resume_id: str
    user_id: str
    source_filename: str
    sections_ingested: list[SectionResult]
    name: Optional[str] = None
    skills: list[str] = []
    work_experience_years: Optional[float] = None
    embedding_model: str
    ingested_at: str


# ---------------------------------------------------------------------------
# Retrieval models
# ---------------------------------------------------------------------------

class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: Optional[int] = Field(default=None, ge=1)
    embedding_model: Optional[str] = None


class CandidateResult(BaseModel):
    user_id: str
    resume_id: str
    source_filename: str
    similarity_score: float
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    work_experience_years: Optional[float] = None
    skills: list[str] = []
    objectives: Optional[str] = None
    matched_sections: list[str] = []


class RetrieveResponse(BaseModel):
    query: str
    k_used: int
    candidates: list[CandidateResult]


# ---------------------------------------------------------------------------
# Error models
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    error: str
    message: str
    detail: Optional[dict] = None


# ---------------------------------------------------------------------------
# Document models
# ---------------------------------------------------------------------------

class DocumentResponse(BaseModel):
    resume_id: Optional[str] = None
    source_filename: str
    uploaded_at: str
    name: Optional[str] = None
    work_experience_years: Optional[float] = None
    skills: list[str] = []
