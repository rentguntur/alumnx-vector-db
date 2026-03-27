from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.config import get_config


class VectorFileStore:
    """
    Dual storage for each knowledge base:

      <kb_name>.npy      — float32 vectors (N, dims)  — compact, fast for KNN
      <kb_name>_ids.npy  — chunk_id strings (N,)
      <kb_name>.jsonl    — one JSON line per chunk     — human-readable text record
                           {chunk_id, resume_id, section_name, source_filename, chunk_text, created_at}

    The .npy files hold only numbers; the .jsonl holds only text.
    Together they are the complete, auditable record of every ingested chunk.
    """

    def __init__(self) -> None:
        self.config = get_config()

    def _ensure_path(self) -> Path:
        path = self.config.vector_store_path
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _vec_path(self, kb_name: str) -> Path:
        return self._ensure_path() / f"{kb_name}.npy"

    def _ids_path(self, kb_name: str) -> Path:
        return self._ensure_path() / f"{kb_name}_ids.npy"

    def _jsonl_path(self, kb_name: str) -> Path:
        return self._ensure_path() / f"{kb_name}.jsonl"

    # ── Vector (.npy) operations ──────────────────────────────────────

    def read(self, kb_name: str) -> tuple[np.ndarray, list[str]]:
        """Return (vectors, chunk_ids). Empty arrays if KB does not exist."""
        vec_path = self._vec_path(kb_name)
        ids_path = self._ids_path(kb_name)
        if not vec_path.exists() or not ids_path.exists():
            return np.empty((0, self.config.vector_size), dtype=np.float32), []
        vectors = np.load(vec_path, mmap_mode="r")
        chunk_ids = np.load(ids_path, allow_pickle=False).tolist()
        return vectors, chunk_ids

    def append(
        self,
        kb_name: str,
        chunk_ids: list[str],
        vectors: np.ndarray,
        text_records: list[dict] | None = None,
    ) -> None:
        """
        Append new chunk_ids and their normalised vectors to the binary files.
        If text_records is provided, also append those lines to the .jsonl file.
        Each record in text_records must have at minimum: chunk_id, resume_id,
        section_name, source_filename, chunk_text, created_at.
        """
        existing_vectors, existing_ids = self.read(kb_name)
        new_vectors = (
            np.vstack([existing_vectors, vectors]).astype(np.float32)
            if existing_ids
            else vectors.astype(np.float32)
        )
        new_ids = existing_ids + chunk_ids
        self._write_npy(kb_name, new_ids, new_vectors)

        if text_records:
            self._append_jsonl(kb_name, text_records)

    def remove_chunk_ids(self, kb_name: str, ids_to_remove: set[str]) -> None:
        """Remove specific chunk_ids from both .npy and .jsonl files."""
        existing_vectors, existing_ids = self.read(kb_name)
        if not existing_ids:
            return
        mask = np.array([cid not in ids_to_remove for cid in existing_ids])
        kept_ids = [cid for cid, keep in zip(existing_ids, mask) if keep]
        kept_vectors = existing_vectors[mask]
        self._write_npy(kb_name, kept_ids, kept_vectors)
        self._remove_jsonl_ids(kb_name, ids_to_remove)

    def list_kb_names(self) -> list[str]:
        path = self._ensure_path()
        return sorted(p.stem for p in path.glob("*.npy") if not p.stem.endswith("_ids"))

    def delete_kb(self, kb_name: str) -> None:
        for p in (self._vec_path(kb_name), self._ids_path(kb_name), self._jsonl_path(kb_name)):
            if p.exists():
                p.unlink()

    # ── JSONL (.jsonl) operations ─────────────────────────────────────

    def read_jsonl(self, kb_name: str) -> list[dict]:
        """Return all active text records from the .jsonl file."""
        path = self._jsonl_path(kb_name)
        if not path.exists():
            return []
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records

    # ── Internal helpers ──────────────────────────────────────────────

    def _write_npy(self, kb_name: str, chunk_ids: list[str], vectors: np.ndarray) -> None:
        np.save(self._vec_path(kb_name), vectors)
        np.save(self._ids_path(kb_name), np.array(chunk_ids, dtype=str))

    def _append_jsonl(self, kb_name: str, records: list[dict]) -> None:
        path = self._jsonl_path(kb_name)
        with path.open("a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _remove_jsonl_ids(self, kb_name: str, ids_to_remove: set[str]) -> None:
        path = self._jsonl_path(kb_name)
        if not path.exists():
            return
        kept_lines: list[str] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("chunk_id") not in ids_to_remove:
                        kept_lines.append(line)
                except json.JSONDecodeError:
                    kept_lines.append(line)
        with path.open("w", encoding="utf-8") as f:
            for line in kept_lines:
                f.write(line + "\n")
