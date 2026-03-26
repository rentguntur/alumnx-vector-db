from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile

from app.config import get_config
from app.errors import error_response
from app.services.ingestion import ingest_file
from app.utils import slugify_name


router = APIRouter()
logger = logging.getLogger("nexvec.ingest")


def _clean_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped or stripped.lower() == "string":
        return None
    return stripped


@router.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    kb_name: str | None = Form(default=None),
    chunking_strategy: str | None = Form(default=None),
    chunk_size: int | None = Form(default=None),
    overlap_size: int | None = Form(default=None),
    embedding_model: str | None = Form(default=None),
    overwrite: bool = Form(default=False),
):
    kb_name = _clean_optional_text(kb_name)
    config = get_config()
    resolved_chunking_strategy = _clean_optional_text(chunking_strategy) or config.default_chunking_strategy
    
    # Swagger often sends '0' as default for integers; treat as None to use config defaults
    effective_chunk_size = chunk_size if (chunk_size is not None and chunk_size > 0) else None
    effective_overlap_size = overlap_size if (overlap_size is not None and overlap_size > 0) else None
    
    embedding_model = _clean_optional_text(embedding_model)
    logger.info(
        "Ingest request received file=%s kb_name=%s chunking_strategy=%s chunk_size=%s overlap_size=%s overwrite=%s embedding_model=%s",
        file.filename,
        kb_name,
        resolved_chunking_strategy,
        effective_chunk_size,
        effective_overlap_size,
        overwrite,
        embedding_model,
    )

    supported_extensions = (".pdf", ".png", ".jpg", ".jpeg", ".mp4", ".mp3", ".wav", ".m4a")
    if not file.filename.lower().endswith(supported_extensions):
        return error_response(400, "INVALID_FILE_TYPE", "Unsupported file type. Supported types: PDF, Images, Audio, Video.", {"source_filename": file.filename})

    resolved_model = embedding_model or config.embedding_model

    ext = Path(file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as handle:
        temp_path = Path(handle.name)
        content = await file.read()
        handle.write(content)
    logger.info("Upload stored temporarily at %s", temp_path)

    try:
        response = ingest_file(
            file_name=file.filename,
            file_path=str(temp_path),
            kb_name=kb_name,
            chunking_strategy=resolved_chunking_strategy,
            chunk_size=effective_chunk_size,
            overlap_size=effective_overlap_size,
            embedding_model=resolved_model,
            overwrite=overwrite,
        )
        logger.info("Ingest completed kb_name=%s strategies=%s", response.kb_name, [item.strategy_name for item in response.strategies_processed])
        return response.model_dump()
    except FileExistsError as exc:
        return error_response(
            409,
            "DUPLICATE_ENTRY",
            "Active chunks already exist for this file, strategy, and model. Pass overwrite=true to replace them.",
            {
                "source_filename": file.filename,
                "chunking_strategy": resolved_chunking_strategy,
                "embedding_model": resolved_model,
                "kb_name": slugify_name(kb_name) if kb_name else None,
            },
        )
    except LookupError:
        return error_response(400, "NO_EXTRACTABLE_TEXT", "No extractable text was found in the PDF.", {"source_filename": file.filename})
    except ValueError as exc:
        message = str(exc)
        if "password-protected" in message.lower() or "encrypted" in message.lower():
            return error_response(400, "INVALID_FILE_TYPE", "PDF is encrypted or password-protected.", {"source_filename": file.filename})
        if "chunking strategy" in message.lower():
            return error_response(400, "INVALID_CHUNKING_STRATEGY", message)
        return error_response(400, "INGESTION_ERROR", message, {"source_filename": file.filename})
    except Exception as exc:
        return error_response(500, "INGESTION_ERROR", str(exc), {"source_filename": file.filename})
    finally:
        logger.info("Cleaning up temporary file %s", temp_path)
        if temp_path.exists():
            temp_path.unlink()
