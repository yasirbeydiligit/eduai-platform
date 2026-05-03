"""TurkishEmbedder LRU query cache testleri (Sapma 36 — Task 4 perf).

FakeEmbedder cache mantığı içermiyor (override edilmiş `embed_query`); bu
testlerde gerçek `TurkishEmbedder` cache logic'ini doğrulamak için
**model load'u atlatılmış mock** kullanılır: `_ensure_loaded` monkeypatch'le
no-op yapılır + `_model.encode` lambda ile.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest

from agents.rag.embeddings import TurkishEmbedder


class _FakeST:
    """Sentence-transformers stub — `encode()` deterministik vektör döndürür.

    Cache testleri için kayıt amaçlı: kaç kez encode çağrıldı?
    """

    def __init__(self, vector_size: int = 8) -> None:
        self.vector_size = vector_size
        self.encode_call_count = 0

    def encode(
        self,
        text,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ):
        self.encode_call_count += 1
        # Deterministik vektör — text uzunluğuna göre farklı.
        h = hash(text if isinstance(text, str) else " ".join(text)) % 1000
        vec = np.array([h / 1000.0] * self.vector_size, dtype=np.float32)
        return vec


@pytest.fixture
def cached_embedder(monkeypatch: pytest.MonkeyPatch) -> tuple[TurkishEmbedder, _FakeST]:
    """Gerçek TurkishEmbedder ama mock SentenceTransformer'la — cache logic test."""
    embedder = TurkishEmbedder(model_id="test/fake", query_cache_size=4)
    fake_model = _FakeST(vector_size=8)
    # Lazy slot manuel doldur — _ensure_loaded re-load yapmasın.
    embedder._model = fake_model  # noqa: SLF001
    embedder._vector_size = 8  # noqa: SLF001
    return embedder, fake_model


def test_cache_hit_skips_encode(cached_embedder) -> None:
    """Aynı sorgu ikinci kez gelirse encode tekrar çağrılmaz."""
    embedder, fake_model = cached_embedder

    v1 = embedder.embed_query("Tanzimat nedir?")
    v2 = embedder.embed_query("Tanzimat nedir?")

    assert v1 == v2, "Aynı sorgu aynı vektörü döndürmeli"
    assert fake_model.encode_call_count == 1, "Cache hit'te encode tekrar çağrılmamalı"
    stats = embedder.cache_stats
    assert stats["hits"] == 1
    assert stats["misses"] == 1


def test_cache_miss_for_different_query(cached_embedder) -> None:
    """Farklı sorgu → cache miss → encode çağrılır."""
    embedder, fake_model = cached_embedder

    embedder.embed_query("Tanzimat nedir?")
    embedder.embed_query("Newton kimdir?")

    assert fake_model.encode_call_count == 2
    assert embedder.cache_stats["misses"] == 2


def test_lru_eviction(cached_embedder) -> None:
    """maxsize=4 → 5. sorgu en eski (LRU) item'ı çıkartmalı.

    Not: `test/fake` model_id'si E5 değil → query_prefix="" → cache key
    direkt sorgu metni.
    """
    embedder, fake_model = cached_embedder

    queries = ["q1", "q2", "q3", "q4", "q5"]
    for q in queries:
        embedder.embed_query(q)

    # Cache 4 itemli olmalı; q1 evict edildi.
    assert len(embedder._query_cache) == 4  # noqa: SLF001
    assert "q1" not in embedder._query_cache  # noqa: SLF001
    # q1'i tekrar sor → cache miss → 6. encode call
    embedder.embed_query("q1")
    assert fake_model.encode_call_count == 6


def test_lru_move_to_end_on_access(cached_embedder) -> None:
    """Cache hit en yeni kullanım → en sona taşınır → LRU eviction sırası değişir."""
    embedder, fake_model = cached_embedder

    embedder.embed_query("a")
    embedder.embed_query("b")
    embedder.embed_query("c")
    embedder.embed_query("d")
    # cache: [a, b, c, d] (en eski → en yeni)
    embedder.embed_query("a")
    # cache: [b, c, d, a] — a en yeni
    embedder.embed_query("e")
    # cache: [c, d, a, e] — b evict (en eskiydi)
    cache_keys = list(embedder._query_cache.keys())  # noqa: SLF001
    # `test/fake` model_id E5 değil → prefix yok → key = sorgu metni.
    assert "b" not in cache_keys, f"b evict edilmeliydi, kaldı: {cache_keys}"
    assert "a" in cache_keys, "a recently accessed → cache'te kalmalı"


def test_cache_disabled_when_size_zero() -> None:
    """query_cache_size=0 → cache devre dışı, her sorgu encode."""
    embedder = TurkishEmbedder(model_id="test/fake", query_cache_size=0)
    fake_model = _FakeST()
    embedder._model = fake_model  # noqa: SLF001
    embedder._vector_size = 8  # noqa: SLF001

    embedder.embed_query("aynı")
    embedder.embed_query("aynı")

    assert fake_model.encode_call_count == 2, "Cache kapalıyken her çağrı encode olmalı"
    assert isinstance(embedder._query_cache, OrderedDict)  # noqa: SLF001
    assert len(embedder._query_cache) == 0  # noqa: SLF001
