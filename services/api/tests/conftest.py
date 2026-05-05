"""Pytest fixtures — tüm test dosyalarında otomatik erişilebilir.

P3 Task 5 entegrasyonu sonrası (Sapma 27 — agents bağımlılığı):
P1 testleri agents/Qdrant'a ihtiyaç duymadan çalışmalı. Lifespan'de
`app.state.indexer` ve `app.state.pipeline` Qdrant down ise None olur →
`get_document_indexer` / `get_agent_pipeline` dependency'leri 503 fırlatır.

Çözüm: dependency override ile bu iki getter'ı **mock**'la. P1'in upload
validation testleri (415/413/201) indexer'a gerçek anlamda dokunmuyor;
mock yeter. Gerçek indexer testi `agents/tests/` altında (in-memory Qdrant +
FakeEmbedder).
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> Iterator[TestClient]:
    """FastAPI TestClient + agents/ dependency mock.

    `with TestClient(app)` context manager pattern lifespan event'lerini
    (startup/shutdown) tetikler. P3+ sonrası lifespan agents/ init dener;
    Qdrant down ise app.state=None olur → endpoint 503. Test'lerde bu
    davranışı **mock** ile bypass et.
    """
    # Import'u fixture içinde tutuyoruz — collect time'da main.py'nin import
    # edilmesinden kaçınır (config/env issue'larını fixture-use anına öteler).
    from app.dependencies import get_agent_pipeline, get_document_indexer
    from app.main import app

    # MagicMock — index_file() çağrısı 5 chunk döndürür (validation testleri
    # için yeter; gerçek indexer logic agents/tests'te test edilir).
    mock_indexer = MagicMock()
    mock_indexer.index_file.return_value = 5  # mock chunks_indexed

    mock_pipeline = MagicMock()

    app.dependency_overrides[get_document_indexer] = lambda: mock_indexer
    app.dependency_overrides[get_agent_pipeline] = lambda: mock_pipeline

    with TestClient(app) as c:
        yield c

    # Cleanup — fixture sonrası override'ları temizle (test izolasyonu).
    app.dependency_overrides.clear()
