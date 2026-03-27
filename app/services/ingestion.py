from __future__ import annotations

import hashlib
import logging
import uuid

import numpy as np

from app.config import get_config
from app.models import IngestResponse, SectionResult
from app.services.embedding.embedder import GeminiEmbedder
from app.services.llm_parser import ParsedResume, parse_resume
from app.services.pdf_extractor import extract_pdf_pages
from app.services.store.postgres_store import PostgresStore
from app.services.store.vector_file_store import VectorFileStore
from app.utils import now_ist_iso

logger = logging.getLogger("nexvec.ingestion")

# Fixed embeddable sections (work_experience_years is numeric — stored in DB, not embedded)
EMBEDDABLE_SECTIONS = [
    "objectives",
    "work_experience_text",
    "projects",
    "education",
    "skills",
    "achievements",
]

UNIVERSAL_VECTOR_STORE = "nex_vec"


def _hash_file(file_path: str) -> str:
    """SHA-256 of file bytes — content-based identity, filename-independent."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha256.update(block)
    return sha256.hexdigest()


def _normalise(arr: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(arr)
    return (arr / norm) if norm > 0 else arr


def _section_text(parsed: ParsedResume, section: str) -> str | None:
    """Return embeddable text for a section. Skills list is joined to a string."""
    if section == "skills":
        return ", ".join(parsed.skills) if parsed.skills else None
    return getattr(parsed, section, None)


def ingest_file(
    file_name: str,
    file_path: str,
    embedding_model: str | None,
) -> IngestResponse:
    """
    Ingest a PDF resume:

      1. SHA-256 content hash — skip if already ingested
      2. Extract text → LLM parses into 7 fixed sections + identity fields
      3. Resolve or create user (email/phone identity)
      4. Embed 6 sections (unit-normalised) → per-section .npy files
      5. Persist resume row with section texts + chunk_ids to Postgres
    """
    config = get_config()
    pg = PostgresStore()
    vfs = VectorFileStore()
    active_model = embedding_model or config.embedding_model

    # ── Content-hash deduplication ────────────────────────────────────
    file_hash = _hash_file(file_path)
    existing_resume_id = pg.get_resume_id_by_hash(file_hash)

    if existing_resume_id:
        logger.info(
            "Duplicate content (hash=%s...), returning existing resume_id=%s",
            file_hash[:16], existing_resume_id,
        )
        resume = pg.get_resume_by_id(existing_resume_id)
        r = resume or {}
        sections_ingested = [
            SectionResult(section_name=s, chunk_id=r[f"{s}_chunk_id"])
            for s in EMBEDDABLE_SECTIONS
            if r.get(f"{s}_chunk_id")
        ]
        return IngestResponse(
            resume_id=existing_resume_id,
            user_id=r.get("user_id", ""),
            source_filename=r.get("source_filename", file_name),
            sections_ingested=sections_ingested,
            name=r.get("name"),
            skills=list(r.get("skills") or []),
            work_experience_years=r.get("work_experience_years"),
            embedding_model=active_model,
            ingested_at=r.get("created_at", now_ist_iso()),
        )

    # ── Extract text ──────────────────────────────────────────────────
    pages = extract_pdf_pages(file_path)
    if not pages:
        raise LookupError("NO_EXTRACTABLE_TEXT")
    full_text = "\n\n".join(p.text for p in pages if p.text.strip())
    logger.info("Extracted %d pages (%d chars) from %s", len(pages), len(full_text), file_name)

    # ── LLM parse → 7 fixed sections + identity ───────────────────────
    parsed = parse_resume(full_text)

    # ── Resolve or create user (email/phone identity) ──────────────────
    user_id = pg.get_user_id_by_contact(parsed.email, parsed.phone) or str(uuid.uuid4())
    created_at = now_ist_iso()

    pg.upsert_user({
        "user_id": user_id,
        "name": parsed.name,
        "email": parsed.email,
        "phone": parsed.phone,
        "location": parsed.location,
        "created_at": created_at,
    })
    logger.info("User: user_id=%s (%s)", user_id,
                "existing" if pg.get_user_id_by_contact(parsed.email, parsed.phone) else "new")

    # ── Embed 6 sections ──────────────────────────────────────────────
    resume_id = str(uuid.uuid4())
    embedder = GeminiEmbedder(active_model)

    # Collect non-empty sections for batch embedding
    to_embed = [
        (section, _section_text(parsed, section))
        for section in EMBEDDABLE_SECTIONS
    ]
    to_embed = [(s, t) for s, t in to_embed if t]

    chunk_ids: dict[str, str] = {}

    if to_embed:
        section_names_to_embed = [s for s, _ in to_embed]
        texts_to_embed = [t for _, t in to_embed]
        raw_vectors = embedder.embed_texts(texts_to_embed)

        all_cids: list[str] = []
        normed_matrix: list[np.ndarray] = []

        for section, raw_vec in zip(section_names_to_embed, raw_vectors):
            cid = str(uuid.uuid4())
            chunk_ids[section] = cid
            normed = _normalise(np.asarray(raw_vec, dtype=np.float32))
            all_cids.append(cid)
            normed_matrix.append(normed)

        # All section vectors for this resume appended to the single flat store
        stacked = np.stack(normed_matrix).astype(np.float32)
        vfs.append(
            UNIVERSAL_VECTOR_STORE,
            all_cids,
            stacked,
            text_records=[
                {"chunk_id": cid, "resume_id": resume_id, "vector": vec.tolist()}
                for cid, vec in zip(all_cids, stacked)
            ],
        )

    # ── Persist resume row ────────────────────────────────────────────
    pg.insert_resume({
        "resume_id": resume_id,
        "user_id": user_id,
        "source_filename": file_name,
        "file_hash": file_hash,
        "objectives": parsed.objectives,
        "work_experience_years": parsed.work_experience_years,
        "work_experience_text": parsed.work_experience_text,
        "projects": parsed.projects,
        "education": parsed.education,
        "skills": parsed.skills or [],
        "achievements": parsed.achievements,
        "objectives_chunk_id": chunk_ids.get("objectives"),
        "work_experience_text_chunk_id": chunk_ids.get("work_experience_text"),
        "projects_chunk_id": chunk_ids.get("projects"),
        "education_chunk_id": chunk_ids.get("education"),
        "skills_chunk_id": chunk_ids.get("skills"),
        "achievements_chunk_id": chunk_ids.get("achievements"),
        "embedding_model": embedder.model,
        "is_active": True,
        "created_at": created_at,
    })

    logger.info(
        "Ingested file=%s resume_id=%s user_id=%s sections_embedded=%d model=%s",
        file_name, resume_id, user_id, len(chunk_ids), embedder.model,
    )

    sections_ingested = [
        SectionResult(section_name=s, chunk_id=chunk_ids[s])
        for s in EMBEDDABLE_SECTIONS
        if s in chunk_ids
    ]
    return IngestResponse(
        resume_id=resume_id,
        user_id=user_id,
        source_filename=file_name,
        sections_ingested=sections_ingested,
        name=parsed.name,
        skills=parsed.skills,
        work_experience_years=parsed.work_experience_years,
        embedding_model=embedder.model,
        ingested_at=created_at,
    )
