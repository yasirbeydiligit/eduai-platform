"""Pytest fixtures — tüm test dosyalarında otomatik erişilebilir."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> Iterator[TestClient]:
    """FastAPI TestClient fixture'ı.

    `with TestClient(app)` context manager pattern lifespan event'lerini (startup/shutdown)
    tetikler. P1'de lifespan body'si boş, ama P2/P3'te DB connection pool init burada olacak;
    bu pattern'i şimdiden doğru kullanmak sonra refactor gerektirmez.
    """
    # Import'u fixture içinde tutuyoruz — collect time'da main.py'nin import edilmesinden
    # kaçınır (config/env issue'larını fixture-use anına öteler).
    from app.main import app

    with TestClient(app) as c:
        yield c
