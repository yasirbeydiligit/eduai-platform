"""Indexer smoke test runner — Task 10 doğrulama.

agents/data/tarih_tanzimat.txt'i indeksle, sonra:
  1. list_documents() ile özetini yazdır
  2. Aynı dosyayı tekrar yükleyerek duplicate-skip davranışını doğrula

Kullanım:
    docker-compose up qdrant -d
    source .venv-agents/bin/activate
    python agents/scripts/index_seed.py
"""

from __future__ import annotations

from pathlib import Path

from agents.rag.indexer import DocumentIndexer

SAMPLE_FILE = Path(__file__).resolve().parents[1] / "data" / "tarih_tanzimat.txt"


def main() -> int:
    print(f"Test dosyası: {SAMPLE_FILE}")
    if not SAMPLE_FILE.exists():
        print(f"  ✗ Dosya bulunamadı: {SAMPLE_FILE}")
        return 1

    print("\n[1/3] Indexer kuruluyor (Qdrant + embedder)...")
    indexer = DocumentIndexer()
    print(
        f"  ✓ Hazır — collection='{indexer.collection_name}', "
        f"vector_size={indexer.embedder.vector_size}"
    )

    print("\n[2/3] Dosyayı indeksle (ilk yükleme)...")
    chunks_first = indexer.index_file(
        SAMPLE_FILE, metadata={"subject": "tarih", "grade_level": 9}
    )
    print(f"  → İlk yükleme: {chunks_first} chunk")

    print("\n[3/3] Aynı dosyayı tekrar indeksle (duplicate-skip beklenen)...")
    chunks_second = indexer.index_file(
        SAMPLE_FILE, metadata={"subject": "tarih", "grade_level": 9}
    )
    print(f"  → İkinci yükleme: {chunks_second} chunk (0 olmalı)")

    print("\nDoküman özeti (list_documents):")
    for doc in indexer.list_documents():
        print(
            f"  • {doc['source']} (subject={doc.get('subject')}) — "
            f"{doc['chunks']} chunk, {doc['page_count']} sayfa, "
            f"doc_id={doc['doc_id']}"
        )

    if chunks_first == 0:
        print("\n  ✗ İlk yükleme 0 chunk üretti (Qdrant'ta zaten var olabilir).")
        print(
            "    Re-test için: collection'ı sıfırla veya farklı QDRANT_COLLECTION kullan."
        )
        return 1
    if chunks_second != 0:
        print("\n  ✗ Duplicate-skip çalışmadı (ikinci yükleme >0 chunk).")
        return 1

    print("\n✓ Indexer smoke test PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
