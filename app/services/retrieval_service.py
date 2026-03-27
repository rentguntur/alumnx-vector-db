from __future__ import annotations

import logging

import numpy as np

from app.config import get_config
from app.models import CandidateResult, RetrieveRequest, RetrieveResponse
from app.services.embedding.embedder import GeminiEmbedder
from app.services.ingestion import EMBEDDABLE_SECTIONS, UNIVERSAL_VECTOR_STORE
from app.services.llm_query import generate_sql_query
from app.services.store.postgres_store import PostgresStore
from app.services.store.vector_file_store import VectorFileStore

logger = logging.getLogger("nexvec.retrieval")


def retrieve_documents(request: RetrieveRequest) -> RetrieveResponse:
    """
    SQL-first semantic search over resumes.

    Flow:
      1. LLM converts query → SQL → Postgres returns matching resume_ids (structured filter)
      2. Fetch chunk_ids for those resumes from the resumes table (no extra lookup needed)
      3. Embed query → cosine similarity ONLY against the filtered resumes' vectors
      4. Deduplicate by user_id — one result per person, best-matching resume wins
      5. Return ranked candidates
    """
    config = get_config()
    k = request.k or config.knn_k
    pg = PostgresStore()
    vfs = VectorFileStore()

    if not request.query.strip():
        raise ValueError("EMPTY_QUERY")

    embedding_model = request.embedding_model or config.embedding_model
    logger.info("Retrieve: query=%r k=%s model=%s", request.query, k, embedding_model)

    # ── Step 1: SQL filter → resume_ids ───────────────────────────────
    sql_failed = False
    try:
        sql = generate_sql_query(request.query)
        resume_ids = pg.execute_sql_query(sql)
    except Exception as exc:
        logger.warning("SQL generation/execution failed (%s), falling back to all resumes", exc)
        resume_ids = []
        sql_failed = True

    if resume_ids:
        resume_rows = pg.get_resumes_by_ids(resume_ids)
    elif sql_failed:
        # SQL errored out — fall back to full scan as a safety net
        resume_rows = pg.get_all_active_resumes()
    else:
        # SQL ran fine but matched nothing — no candidates meet the criteria
        logger.info("SQL filter returned no matches for query=%r", request.query)
        return RetrieveResponse(query=request.query, k_used=k, candidates=[])

    if not resume_rows:
        logger.info("No active resumes found")
        return RetrieveResponse(query=request.query, k_used=k, candidates=[])

    # ── Step 2: Build chunk_id → resume/section mapping ──────────────
    chunk_to_resume: dict[str, str] = {}
    chunk_to_section: dict[str, str] = {}

    for row in resume_rows:
        for section in EMBEDDABLE_SECTIONS:
            cid = row.get(f"{section}_chunk_id")
            if cid:
                chunk_to_resume[cid] = row["resume_id"]
                chunk_to_section[cid] = section

    # ── Step 3: Embed query (unit-normalised) ─────────────────────────
    embedder = GeminiEmbedder(embedding_model)
    query_vector = np.asarray(embedder.embed_query(request.query), dtype=np.float32)
    norm = np.linalg.norm(query_vector)
    if norm > 0:
        query_vector = query_vector / norm

    # ── Step 4: Targeted vector search on flat store ──────────────────
    # Load the single flat file once, filter to SQL-filtered chunks only.
    # Dot product = cosine similarity since all vectors are unit-normalised.
    all_vectors, all_ids = vfs.read(UNIVERSAL_VECTOR_STORE)
    score_by_chunk: dict[str, float] = {}

    if len(all_ids) > 0:
        id_to_pos = {cid: i for i, cid in enumerate(all_ids)}
        target_chunks = [cid for cid in chunk_to_resume if cid in id_to_pos]

        if target_chunks:
            positions = [id_to_pos[cid] for cid in target_chunks]
            subset = all_vectors[positions]
            scores = (subset @ query_vector).tolist()
            for cid, score in zip(target_chunks, scores):
                score_by_chunk[cid] = float(score)

    if not score_by_chunk:
        return RetrieveResponse(query=request.query, k_used=k, candidates=[])

    # ── Step 5: Best score + matched sections per resume ──────────────
    best_score: dict[str, float] = {}
    matched_sections: dict[str, list[str]] = {}

    for cid, score in score_by_chunk.items():
        rid = chunk_to_resume[cid]
        section = chunk_to_section[cid]
        if rid not in best_score or score > best_score[rid]:
            best_score[rid] = score
        matched_sections.setdefault(rid, [])
        if section not in matched_sections[rid]:
            matched_sections[rid].append(section)

    ranked_ids = sorted(best_score, key=lambda r: best_score[r], reverse=True)

    # ── Step 6: Deduplicate by user_id ────────────────────────────────
    resume_rows_by_id = {r["resume_id"]: r for r in resume_rows}
    seen_users: set[str] = set()
    deduped_ids: list[str] = []

    for rid in ranked_ids:
        uid = resume_rows_by_id.get(rid, {}).get("user_id") or rid
        if uid not in seen_users:
            seen_users.add(uid)
            deduped_ids.append(rid)
        if len(deduped_ids) == k:
            break

    # ── Step 7: Build response ────────────────────────────────────────
    candidates: list[CandidateResult] = []
    for rid in deduped_ids:
        row = resume_rows_by_id.get(rid, {})
        candidates.append(CandidateResult(
            user_id=row.get("user_id", rid),
            resume_id=rid,
            source_filename=row.get("source_filename", ""),
            similarity_score=round(best_score[rid], 6),
            name=row.get("name"),
            email=row.get("email"),
            phone=row.get("phone"),
            location=row.get("location"),
            work_experience_years=row.get("work_experience_years"),
            skills=list(row.get("skills") or []),
            objectives=row.get("objectives"),
            matched_sections=matched_sections.get(rid, []),
        ))

    logger.info(
        "Retrieve complete: %d unique candidates from %d resumes",
        len(candidates), len(resume_rows),
    )
    return RetrieveResponse(query=request.query, k_used=k, candidates=candidates)
