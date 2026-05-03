"""RAG akış testleri — indexer + retriever + subject filter + boş durum.

Mock'lar conftest.py'de: in-memory Qdrant + FakeEmbedder (keyword-bazlı).
Bu testler **akış doğruluğu**na odaklı; semantik kalite gerçek e5-large'la
empirical olarak smoke test'lerde doğrulandı (Task 2 score 0.90+).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.rag.indexer import DocumentIndexer
from agents.rag.retriever import EduRetriever


def _write_text(tmp_path: Path, name: str, content: str) -> Path:
    """Test data helper — tmp_path'e .txt dosya yaz."""
    file_path = tmp_path / name
    file_path.write_text(content, encoding="utf-8")
    return file_path


def test_index_and_retrieve(
    indexer: DocumentIndexer,
    retriever: EduRetriever,
    tmp_path: Path,
) -> None:
    """Küçük metin yükle → retrieve et → en az 1 sonuç döner ve içeriği uyumlu."""
    file_path = _write_text(
        tmp_path,
        "tanzimat.txt",
        "Tanzimat Fermanı 1839'da Sultan Abdülmecid döneminde ilan edildi. "
        "Ferman modernleşme sürecinin başlangıcıdır.",
    )

    chunks = indexer.index_file(file_path, metadata={"subject": "tarih"})
    assert chunks > 0, "Indexer en az 1 chunk yüklemeli"

    docs = retriever.retrieve("Tanzimat ne zaman ilan edildi?", k=4)
    assert len(docs) > 0, "Retriever en az 1 sonuç döndürmeli"

    top = docs[0]
    # FakeEmbedder keyword-bazlı; "tanzimat" hem sorgu hem chunk içinde → match.
    assert "Tanzimat" in top.page_content, "Top chunk Tanzimat içermeli"
    assert top.metadata["source"] == "tanzimat.txt"
    assert top.metadata["subject"] == "tarih"
    assert top.metadata["score"] > 0.0, "Score metadata pozitif olmalı"


def test_retrieve_with_subject_filter(
    indexer: DocumentIndexer,
    retriever: EduRetriever,
    tmp_path: Path,
) -> None:
    """Subject filter: tarih sorgusunda sadece tarih chunk'ları, fizik benzer."""
    tarih_file = _write_text(
        tmp_path,
        "tarih.txt",
        "Tanzimat Fermanı 1839 modernleşme tarih.",
    )
    fizik_file = _write_text(
        tmp_path,
        "fizik.txt",
        "Newton hareket yasaları kuvvet ivme eylemsizlik fizik.",
    )

    indexer.index_file(tarih_file, metadata={"subject": "tarih"})
    indexer.index_file(fizik_file, metadata={"subject": "fizik"})

    # Tarih filter — sadece tarih dokümanından sonuç beklenir.
    tarih_docs = retriever.retrieve("herhangi bir konu", subject="tarih", k=4)
    assert len(tarih_docs) > 0
    for doc in tarih_docs:
        assert doc.metadata["subject"] == "tarih", (
            f"Subject filter başarısız: {doc.metadata}"
        )

    # Fizik filter — sadece fizik dokümanı.
    fizik_docs = retriever.retrieve("herhangi bir konu", subject="fizik", k=4)
    assert len(fizik_docs) > 0
    for doc in fizik_docs:
        assert doc.metadata["subject"] == "fizik"

    # Bilinmeyen subject → boş sonuç.
    none_docs = retriever.retrieve("herhangi", subject="kimya", k=4)
    assert none_docs == [], "Olmayan subject için boş liste beklenir"


def test_empty_retrieve(retriever: EduRetriever) -> None:
    """Boş collection → retriever boş liste döner (exception fırlatmaz)."""
    docs = retriever.retrieve("herhangi bir soru", k=4)
    assert docs == [], "Boş collection'dan boş liste beklenir"


def test_duplicate_skip(
    indexer: DocumentIndexer,
    tmp_path: Path,
) -> None:
    """Aynı dosya iki kez yüklenirse ikinci sefer chunks=0 (Sapma 32 fix testi)."""
    file_path = _write_text(tmp_path, "doc.txt", "Tanzimat 1839 ferman tarih.")

    first = indexer.index_file(file_path, metadata={"subject": "tarih"})
    second = indexer.index_file(file_path, metadata={"subject": "tarih"})

    assert first > 0, "İlk yükleme chunk yüklemeli"
    assert second == 0, "İkinci yükleme duplicate-skip → 0 chunk"


def test_source_name_override(
    indexer: DocumentIndexer,
    tmp_path: Path,
) -> None:
    """source_name parametresi doc_id stem'ini etkiler (Sapma 32).

    Aynı içerik, farklı physical path, aynı source_name → duplicate-skip.
    Aynı içerik, aynı physical path, farklı source_name → re-index.
    """
    content = "Tanzimat 1839 ferman tarih."
    p1 = _write_text(tmp_path, "tmp_random_name.txt", content)
    p2 = _write_text(tmp_path, "another_random.txt", content)

    # Her ikisini source_name="canonical.txt" ile indeksle → ikincisi skip
    first = indexer.index_file(
        p1, metadata={"subject": "tarih"}, source_name="canonical.txt"
    )
    second = indexer.index_file(
        p2, metadata={"subject": "tarih"}, source_name="canonical.txt"
    )

    assert first > 0
    assert second == 0, "Aynı source_name + aynı içerik → duplicate-skip beklenir"


@pytest.mark.parametrize("k", [1, 2, 4])
def test_retrieve_k_parameter(
    indexer: DocumentIndexer,
    retriever: EduRetriever,
    tmp_path: Path,
    k: int,
) -> None:
    """retrieve(k=N) en fazla N sonuç döndürmeli."""
    # 4 farklı paragraf → splitter 4+ chunk üretir
    content = "\n\n".join(
        [
            "Tanzimat 1839 ferman.",
            "Newton hareket yasaları.",
            "Mehmet Akif şair edebiyat.",
            "Matematik ivme kuvvet.",
        ]
    )
    file_path = _write_text(tmp_path, "mixed.txt", content)
    indexer.index_file(file_path, metadata={"subject": "genel"})

    docs = retriever.retrieve("herhangi", k=k)
    assert len(docs) <= k, f"k={k} ama {len(docs)} sonuç döndü"
