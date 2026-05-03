"""pytest fixture'ları — in-memory Qdrant + fake embedder + mock LLM.

Test izolasyon stratejisi:
- **In-memory Qdrant**: `QdrantClient(":memory:")` qdrant-client local mode
  → ağ/disk yok, test arası clean state.
- **FakeEmbedder**: keyword-bazlı 16-dim vektör. Gerçek e5-large yüklemek
  CI'da çok ağır (2 GB) ve test izolasyonunu bozar (HF cache yan etki).
  Sapma 34: deterministik fake embedder akış testleri için yeterli.
- **MockLLM**: `LLMBackend` protocol uygulaması; sırayla response döndürür.
- **Retriever singleton reset** (autouse): test arası
  `agents.graph.nodes._retriever_singleton = None` → singleton sızması yok.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable

import pytest
from qdrant_client import QdrantClient

from agents.rag.indexer import (
    DocumentIndexer,
)
from agents.rag.retriever import EduRetriever

# Fake embedder anahtar sözlüğü — test korpusunda geçen Türkçe kelimeler.
# Boyut = sözlük uzunluğu; Qdrant collection vector_size buradan türetilir.
# 16 keyword → 16-dim binary-then-normalized vektör. Cosine similarity'de
# benzer keyword içeren metinler yakın çıkar (semantik kaba ama akış testi
# için yeterli).
_FAKE_KEYWORDS: tuple[str, ...] = (
    "tanzimat",
    "ferman",
    "1839",
    "abdülmecid",
    "modernleşme",
    "newton",
    "fizik",
    "kuvvet",
    "ivme",
    "eylemsizlik",
    "tarih",
    "matematik",
    "edebiyat",
    "şair",
    "yasa",
    "hareket",
)
_FAKE_VECTOR_SIZE = len(_FAKE_KEYWORDS)


class FakeEmbedder:
    """Test için keyword-bazlı sahte embedder. TurkishEmbedder protokolüne uyar."""

    def __init__(self) -> None:
        self.model_id = "fake-keyword-embedder"
        self.device = "cpu"
        # E5 prefix kuralları test'te no-op; FakeEmbedder bağlam-bilmez.
        self.query_prefix = ""
        self.passage_prefix = ""

    @property
    def vector_size(self) -> int:
        return _FAKE_VECTOR_SIZE

    @staticmethod
    def _encode_one(text: str) -> list[float]:
        """Tek metin → 16-dim L2-normalize edilmiş vektör.

        Vektör = her keyword için 1.0 (var) veya 0.0 (yok), sonra normalize.
        Hiç keyword yoksa sıfır-vektör (Qdrant cosine'de undefined; ama test
        senaryolarımızda en az 1 match'leşir, sorun değil).
        """
        lowered = text.lower()
        raw = [1.0 if kw in lowered else 0.0 for kw in _FAKE_KEYWORDS]
        norm = sum(x * x for x in raw) ** 0.5
        if norm == 0:
            return raw
        return [x / norm for x in raw]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._encode_one(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._encode_one(query)


class MockLLM:
    """LLMBackend protocol implementation; sıralı response stub.

    `responses` listesinden sırayla döndürür; liste tükenirse son cevabı
    tekrar verir (sonsuz retry test'i için).
    """

    def __init__(self, responses: Iterable[str]) -> None:
        self.responses: list[str] = list(responses)
        if not self.responses:
            raise ValueError("MockLLM en az bir response gerektirir.")
        self.call_count = 0

    async def generate(self, question: str, context: str, max_tokens: int = 512) -> str:
        idx = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        return self.responses[idx]


# ----- Fixture'lar -----


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    """Keyword-bazlı sahte embedder."""
    return FakeEmbedder()


@pytest.fixture
def in_memory_qdrant() -> QdrantClient:
    """qdrant-client `:memory:` mode — ağ/disk yok, test arası izole."""
    # Not: QdrantClient(":memory:") FastEmbed/local mode başlatır; her fixture
    # call'unda yeni client → temiz state. close() çağırmıyoruz çünkü local
    # mode'un explicit cleanup'ı yok (process'le birlikte gider).
    return QdrantClient(":memory:")


@pytest.fixture
def indexer(
    fake_embedder: FakeEmbedder, in_memory_qdrant: QdrantClient
) -> DocumentIndexer:
    """DocumentIndexer with fake embedder + in-memory client.

    Her test fixture'unda unique collection adı → testler arası çakışma yok.
    """
    collection_name = f"test_{uuid.uuid4().hex[:8]}"
    return DocumentIndexer(
        qdrant_url=":memory:",  # cosmetic — gerçek yol client DI'dan
        collection_name=collection_name,
        embedder=fake_embedder,
        client=in_memory_qdrant,
    )


@pytest.fixture
def retriever(indexer: DocumentIndexer, fake_embedder: FakeEmbedder) -> EduRetriever:
    """EduRetriever — indexer'ın collection + client'ını paylaşır.

    Aynı in-memory client'ı kullanmak şart; aksi halde retriever boş
    collection'a bakar.
    """
    return EduRetriever(
        qdrant_url=":memory:",
        collection_name=indexer.collection_name,
        embedder=fake_embedder,
        client=indexer.client,
    )


@pytest.fixture(autouse=True)
def reset_retriever_singleton():
    """Test arası `agents.graph.nodes._retriever_singleton`'ı sıfırla.

    nodes.py module-level cache → bir test'te set edilen retriever sonraki
    test'e sızmasın. autouse: her test'te otomatik çalışır.
    """
    import agents.graph.nodes as nodes_module

    nodes_module._retriever_singleton = None
    yield
    nodes_module._retriever_singleton = None
