from __future__ import annotations

import logging

from fastapi import APIRouter

from app.errors import error_response
from app.models import RetrieveRequest
from app.services.retrieval_service import retrieve_documents


router = APIRouter()
logger = logging.getLogger("nexvec.retrieve")


def _clean_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped or stripped.lower() == "string":
        return None
    return stripped


@router.post("/retrieve")
def retrieve(request: RetrieveRequest):
    try:
        request.embedding_model = _clean_optional_text(request.embedding_model)
        logger.info(
            "Retrieve request: query=%r k=%s embedding_model=%s",
            request.query, request.k, request.embedding_model,
        )
        response = retrieve_documents(request)
        logger.info("Retrieve completed: %d candidates", len(response.candidates))
        return response.model_dump()
    except ValueError as exc:
        message = str(exc)
        if message == "EMPTY_QUERY":
            return error_response(400, "EMPTY_QUERY", "The query cannot be empty or whitespace only.")
        return error_response(400, "INVALID_REQUEST", message)
    except Exception as exc:
        logger.exception("Retrieval failed")
        return error_response(500, "RETRIEVAL_ERROR", str(exc))
