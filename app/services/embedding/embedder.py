from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.config import get_config

logger = logging.getLogger("nexvec.embedder")

# Extension → MIME type mapping for supported media files
_MIME_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".pdf": "application/pdf",
}


class GeminiEmbedder:
    """Multimodal embedder using the google-genai SDK with gemini-embedding-2-preview."""

    def __init__(self, model: str | None = None) -> None:
        self.config = get_config()
        self.model = model or self.config.embedding_model

    def _client(self):
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is required for embedding. "
                "Install it with: pip install google-genai"
            ) from exc
        return genai.Client()

    # ── Text embedding ────────────────────────────────────────────────

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of text strings, batching as needed."""
        if not texts:
            return []
        client = self._client()
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 100):
            batch = texts[start : start + 100]
            result = client.models.embed_content(
                model=self.model,
                contents=batch,
                config={"output_dimensionality": self.config.output_dimensionality},
            )
            vectors.extend([emb.values for emb in result.embeddings])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        client = self._client()
        result = client.models.embed_content(
            model=self.model,
            contents=text,
            config={"output_dimensionality": self.config.output_dimensionality},
        )
        return result.embeddings[0].values

    # ── Multimodal embedding (images, audio, video, PDF) ──────────────

    def embed_file(self, file_path: str, mime_type: str | None = None) -> list[float]:
        """
        Embed a media file (image, audio, video, or PDF) directly using
        gemini-embedding-2-preview's native multimodal support.

        Returns a single embedding vector for the file.
        """
        try:
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError("google-genai is required for multimodal embedding") from exc

        path = Path(file_path)
        if mime_type is None:
            mime_type = _MIME_TYPES.get(path.suffix.lower())
            if mime_type is None:
                raise ValueError(f"Cannot determine MIME type for extension: {path.suffix}")

        logger.info("Embedding file=%s mime_type=%s model=%s", path.name, mime_type, self.model)
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        client = self._client()
        part = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
        result = client.models.embed_content(
            model=self.model,
            contents=[part],
            config={"output_dimensionality": self.config.output_dimensionality},
        )
        return result.embeddings[0].values
