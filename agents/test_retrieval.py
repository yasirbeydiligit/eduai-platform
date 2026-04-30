"""EduRetriever smoke test — Task 2 doğrulama (TASKS.md gereksinimleri).

Kullanım:
    docker-compose up qdrant -d
    source .venv-agents/bin/activate
    python -m agents.test_retrieval

Önkoşul: agents/scripts/index_seed.py çalıştırılmış (collection dolu).

Test akışı:
    1. Genel retrieve (subject=None) — Tanzimat sorusu
    2. Subject filter test — subject="tarih" aynı sonuç vermeli
    3. Yanlış subject filter — subject="fizik" boş sonuç vermeli
    4. get_context_string çıktısını yazdır
"""

from __future__ import annotations

from agents.rag.retriever import EduRetriever


def _print_results(label: str, docs) -> None:
    """Test bölümü başlığı + her doküman için skor + önizleme + kaynak."""
    print(f"\n--- {label} ({len(docs)} sonuç) ---")
    if not docs:
        print("  (boş sonuç)")
        return
    for i, doc in enumerate(docs, start=1):
        score = doc.metadata.get("score", 0.0)
        source = doc.metadata.get("source", "?")
        page = doc.metadata.get("page", "?")
        # İlk 100 karakter — TASKS.md gereksinimi.
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"  [{i}] score={score:.4f}  source={source}  sayfa={page}")
        print(f"      «{preview}...»")


def main() -> int:
    print("EduRetriever smoke test")
    print(
        "Önkoşul: agents/scripts/index_seed.py ile tarih_tanzimat.txt indekslenmiş olmalı.\n"
    )

    retriever = EduRetriever()
    print(f"Collection: {retriever.collection_name}")
    print(f"Embedder: {retriever.embedder.model_id}")

    query = "Tanzimat Fermanı ne zaman çıktı?"
    print(f"\nSoru: {query!r}")

    # 1) Genel retrieve — subject filtresi yok
    docs_all = retriever.retrieve(query, k=4)
    _print_results("Genel retrieve (subject=None, k=4)", docs_all)

    # 2) Subject filter — Tanzimat dokümanı subject="tarih" ile yüklendi
    docs_history = retriever.retrieve(query, subject="tarih", k=4)
    _print_results("Subject filter: tarih (k=4)", docs_history)

    # 3) Yanlış subject — boş sonuç beklenir (collection'da fizik dokümanı yok)
    docs_physics = retriever.retrieve(query, subject="fizik", k=4)
    _print_results("Subject filter: fizik (boş beklenen)", docs_physics)

    # 4) Context string — RAG prompt formatı
    print("\n--- get_context_string(docs_all) ---")
    print(retriever.get_context_string(docs_all))

    # Doğrulama — minimum sınırlar
    if not docs_all:
        print(
            "\n  ✗ Genel retrieve boş; Qdrant'ta veri yok mu? "
            "agents/scripts/index_seed.py çalıştır."
        )
        return 1
    if len(docs_history) != len(docs_all):
        # Tarih filtresi tüm dokümanlarımıza uyar; aynı sayı beklenir.
        print(
            "\n  ✗ Subject='tarih' filtresi sonuç sayısını değiştirdi "
            "(beklenmeyen). Indexer metadata'sını kontrol et."
        )
        return 1
    if docs_physics:
        print("\n  ✗ Subject='fizik' boş olmalıydı; subject filter çalışmıyor.")
        return 1
    if docs_all[0].metadata.get("score", 0) < 0.7:
        # E5-large benchmark'ta avg 0.86; 0.7 altı ham veri/embedding sorunu.
        print(
            f"\n  ⚠ Top score düşük ({docs_all[0].metadata['score']:.3f}). "
            "Embedding kalitesi kontrol et."
        )
        # Hata değil — uyarı, geçerli olabilir.

    print("\n✓ Retriever smoke test PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
