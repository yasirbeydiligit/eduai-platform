"""Indexer smoke test runner — Task 1 doğrulama + Task 4 multi-disciplinary korpus.

Birden fazla seed dosyasını uygun subject metadata'sıyla indeksler:
  - tarih_tanzimat.txt    → subject="tarih"
  - fizik_newton.txt      → subject="fizik" (Task 4 için)

Aynı dosya re-run'da duplicate-skip davranışı (idempotent) korunur.

Kullanım:
    docker-compose up qdrant -d
    source .venv-agents/bin/activate
    PYTHONPATH=. python agents/scripts/index_seed.py
"""

from __future__ import annotations

from pathlib import Path

from agents.rag.indexer import DocumentIndexer

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# (dosya_adı, subject) çiftleri — yeni doküman eklenince buraya satır eklenir.
SEED_FILES: tuple[tuple[str, str], ...] = (
    ("tarih_tanzimat.txt", "tarih"),
    ("fizik_newton.txt", "fizik"),
)


def main() -> int:
    print(f"Veri dizini: {DATA_DIR}\n")

    indexer = DocumentIndexer()
    print(
        f"  ✓ Indexer hazır — collection='{indexer.collection_name}', "
        f"vector_size={indexer.embedder.vector_size}\n"
    )

    total_indexed = 0
    for filename, subject in SEED_FILES:
        file_path = DATA_DIR / filename
        if not file_path.exists():
            print(f"  ⚠ Atlanıyor — dosya yok: {file_path}")
            continue

        print(f"--- {filename} (subject={subject}) ---")
        chunks = indexer.index_file(
            file_path, metadata={"subject": subject, "grade_level": 9}
        )
        total_indexed += chunks
        print()

    print("Doküman özeti (list_documents):")
    for doc in indexer.list_documents():
        print(
            f"  • {doc['source']} (subject={doc.get('subject')}) — "
            f"{doc['chunks']} chunk, {doc['page_count']} sayfa, "
            f"doc_id={doc['doc_id']}"
        )

    if total_indexed == 0:
        # Tüm dosyalar duplicate olabilir (idempotent re-run); hata değil.
        print("\n  ℹ Yeni chunk yüklenmedi (hepsi zaten indeksli olabilir).")

    print("\n✓ Seed indeksleme tamamlandı")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
