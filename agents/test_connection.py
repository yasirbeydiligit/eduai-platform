"""Qdrant smoke test — P3 Task 0 sonu doğrulama.

Amaç: Qdrant container ayakta mı, qdrant-client lokal kurulumu doğru mu,
collection lifecycle (create → upsert → search → delete) sorunsuz mu?

Kullanım:
    docker-compose up qdrant -d
    python agents/test_connection.py

Beklenen çıktı: tüm adımların ✓ ile tamamlanması ve "Smoke test PASSED" satırı.
İlk başarısız adımda hata mesajı + non-zero exit (CI-friendly).

Not: Bu script production env'inde değil, geliştirme doğrulaması için.
Task 1 indexer'ı qdrant-client'ı zaten kullanacak; bu sadece "altyapı bağlandı"
sinyali. Çalışmazsa Task 1'e geçme.
"""

from __future__ import annotations

import os
import random
import sys
import uuid

# qdrant-client v1.12+ — agents/requirements.txt floor pin
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# ENV'den oku, lokal default — Docker network'te http://qdrant:6333 olur,
# host makinasında http://localhost:6333. Smoke test host'tan çalışıyor.
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# Smoke test boyutu — gerçek embedding modeli Task 1'de seçilecek.
# 384 sentence-transformers MiniLM ailesinin yaygın boyutu; sadece dummy.
VECTOR_SIZE = 384

# Test collection adı — uuid eklenmiş, paralel çalışmalarda çakışma riski yok.
COLLECTION_NAME = f"smoke_test_{uuid.uuid4().hex[:8]}"


def _step(message: str) -> None:
    """Adım başlık formatı — okunabilir konsol çıktısı."""
    print(f"  → {message}", flush=True)


def _ok(message: str) -> None:
    """Başarılı adım sinyali."""
    print(f"  ✓ {message}", flush=True)


def main() -> int:
    """Qdrant lifecycle smoke test'i.

    Returns:
        0 başarılı, 1 başarısız (CI exit code uyumlu).
    """
    print(f"Qdrant smoke test — endpoint: {QDRANT_URL}")
    print(f"Test collection: {COLLECTION_NAME}\n")

    # Adım 1 — bağlantı. timeout=10 sn: Qdrant healthy değilse hızlıca düşsün.
    _step("Qdrant'a bağlanılıyor")
    try:
        client = QdrantClient(url=QDRANT_URL, timeout=10.0)
        # get_collections() en hafif "ping": HTTP 200 + JSON döner.
        existing = client.get_collections()
        _ok(f"Bağlantı kuruldu. Mevcut collection sayısı: {len(existing.collections)}")
    except Exception as exc:
        print(f"  ✗ Bağlantı başarısız: {exc}", file=sys.stderr)
        print(
            "    Kontrol et: 'docker-compose up qdrant -d' çalıştırıldı mı? "
            f"Endpoint erişilebilir mi? ({QDRANT_URL}/readyz)",
            file=sys.stderr,
        )
        return 1

    # Adım 2 — collection oluştur. recreate_collection deprecated; create + var
    # ise sil pattern'i daha açık.
    _step("Test collection oluşturuluyor")
    try:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qmodels.VectorParams(
                size=VECTOR_SIZE,
                # COSINE: sentence-transformers default; vektörler normalize edildiğinde
                # DOT product ile aynı sonuç ama API'da daha açık niyet.
                distance=qmodels.Distance.COSINE,
            ),
        )
        _ok(f"Collection oluşturuldu (vector_size={VECTOR_SIZE}, distance=COSINE)")
    except Exception as exc:
        print(f"  ✗ Collection oluşturulamadı: {exc}", file=sys.stderr)
        return 1

    # Adım 3 — upsert. 3 dummy vektör, payload'ında source bilgisi (gerçek
    # indexer'ın metadata pattern'inin minik prototipi).
    _step("3 dummy vektör yükleniyor")
    try:
        # random.seed → deterministik test; her çalıştırmada aynı vektörler.
        random.seed(42)
        points = [
            qmodels.PointStruct(
                id=i,
                vector=[random.uniform(-1.0, 1.0) for _ in range(VECTOR_SIZE)],
                payload={"source": f"dummy_{i}.txt", "chunk_index": i},
            )
            for i in range(3)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
        # count() upsert'in idempotent görünmesini doğrular.
        count = client.count(collection_name=COLLECTION_NAME, exact=True).count
        if count != 3:
            raise RuntimeError(f"Beklenen 3 nokta, gelen {count}")
        _ok(f"3 vektör upsert edildi, doğrulandı (count={count})")
    except Exception as exc:
        print(f"  ✗ Upsert hatası: {exc}", file=sys.stderr)
        client.delete_collection(COLLECTION_NAME)
        return 1

    # Adım 4 — query. İlk noktayı sorgu vektörü olarak kullan; en yakın
    # komşu kendisi olmalı (similarity ~ 1.0).
    _step("Similarity search test (query_points)")
    try:
        query_vec = points[0].vector
        # query_points qdrant-client v1.10+ tercih edilen API; deprecated `search`
        # yerine.
        result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            limit=1,
        )
        if not result.points:
            raise RuntimeError("Hiç sonuç dönmedi")
        top = result.points[0]
        if top.id != 0:
            raise RuntimeError(f"En yakın komşu id=0 olmalıydı, geldi id={top.id}")
        _ok(f"Search başarılı (top_id={top.id}, score={top.score:.4f})")
    except Exception as exc:
        print(f"  ✗ Search hatası: {exc}", file=sys.stderr)
        client.delete_collection(COLLECTION_NAME)
        return 1

    # Adım 5 — cleanup. Smoke test artığı bırakma; idempotent re-run güvenli.
    _step("Test collection siliniyor")
    try:
        client.delete_collection(COLLECTION_NAME)
        _ok("Collection silindi")
    except Exception as exc:
        print(f"  ✗ Cleanup başarısız: {exc}", file=sys.stderr)
        return 1

    print("\n  Smoke test PASSED — Qdrant altyapısı Task 1 için hazır.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
