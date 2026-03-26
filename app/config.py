from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AppConfig:
    chunk_size: int
    overlap_size: int
    default_chunking_strategy: str
    max_paragraph_size: int
    knn_k: int
    default_retrieval_strategy: dict[str, str]
    embedding_model: str
    output_dimensionality: int
    vector_size: int
    vector_store_path: Path
    min_page_text_length: int
    # Document storage (CRUD upload feature)
    document_store_path: Path
    metadata_store_type: str
    db_host: str
    db_port: str
    db_name: str
    db_user: str
    db_password: str


_CONFIG_CACHE: AppConfig | None = None


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_raw_config() -> dict[str, Any]:
    config_path = project_root() / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def get_config() -> AppConfig:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    raw = _load_raw_config()
    vector_store_path = Path(raw.get("vector_store_path", "./vector_store/"))
    if not vector_store_path.is_absolute():
        vector_store_path = (project_root() / vector_store_path).resolve()

    raw_doc_path = os.getenv("DOCUMENT_STORE_PATH", raw.get("document_store_path", "./document_store/"))
    document_store_path = Path(raw_doc_path)
    if not document_store_path.is_absolute():
        document_store_path = (project_root() / document_store_path).resolve()

    _CONFIG_CACHE = AppConfig(
        chunk_size=int(raw["chunk_size"]),
        overlap_size=int(raw["overlap_size"]),
        default_chunking_strategy=str(raw["default_chunking_strategy"]),
        max_paragraph_size=int(raw["max_paragraph_size"]),
        knn_k=int(raw["knn_k"]),
        default_retrieval_strategy=dict(raw["default_retrieval_strategy"]),
        embedding_model=str(raw["embedding_model"]),
        output_dimensionality=int(raw["output_dimensionality"]),
        vector_size=int(raw["vector_size"]),
        vector_store_path=vector_store_path,
        min_page_text_length=int(raw["min_page_text_length"]),
        document_store_path=document_store_path,
        metadata_store_type=os.getenv("METADATA_STORE_TYPE", raw.get("metadata_store_type", "jsonl")),
        db_host=os.getenv("DB_HOST", "localhost"),
        db_port=os.getenv("DB_PORT", "5432"),
        db_name=os.getenv("DB_NAME", "postgres"),
        db_user=os.getenv("DB_USER", "postgres"),
        db_password=os.getenv("DB_PASSWORD", ""),
    )
    return _CONFIG_CACHE

