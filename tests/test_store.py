from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from app.services.store.vector_file_store import VectorFileStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    cfg = SimpleNamespace(
        vector_store_path=tmp_path / "vector_store",
        vector_size=3,
    )
    monkeypatch.setattr("app.services.store.vector_file_store.get_config", lambda: cfg)
    return VectorFileStore()


def test_read_empty_kb_returns_empty(store):
    vectors, chunk_ids = store.read("nonexistent")
    assert len(chunk_ids) == 0
    assert vectors.shape == (0, 3)


def test_append_and_read(store):
    vecs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    store.append("kb1", ["id-a", "id-b"], vecs)

    vectors, chunk_ids = store.read("kb1")
    assert chunk_ids == ["id-a", "id-b"]
    assert vectors.shape == (2, 3)
    np.testing.assert_allclose(vectors[0], [1.0, 0.0, 0.0])


def test_append_grows_file(store):
    vecs1 = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    vecs2 = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    store.append("kb1", ["id-a"], vecs1)
    store.append("kb1", ["id-b"], vecs2)

    vectors, chunk_ids = store.read("kb1")
    assert len(chunk_ids) == 2
    assert chunk_ids == ["id-a", "id-b"]



def test_remove_chunk_ids(store):
    vecs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    store.append("kb1", ["a", "b", "c"], vecs)

    store.remove_chunk_ids("kb1", {"b"})

    vectors, chunk_ids = store.read("kb1")
    assert chunk_ids == ["a", "c"]
    assert vectors.shape == (2, 3)


def test_list_kb_names(store):
    store.append("kb_alpha", ["x"], np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
    store.append("kb_beta", ["y"], np.array([[0.0, 1.0, 0.0]], dtype=np.float32))

    names = store.list_kb_names()
    assert "kb_alpha" in names
    assert "kb_beta" in names
    # _ids files must not appear as KB names
    assert "kb_alpha_ids" not in names


def test_delete_kb(store):
    store.append("kb1", ["id"], np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
    store.delete_kb("kb1")

    vectors, chunk_ids = store.read("kb1")
    assert len(chunk_ids) == 0
