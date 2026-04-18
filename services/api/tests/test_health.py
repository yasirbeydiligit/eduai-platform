"""GET /health endpoint'i için smoke test."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_ok(client: TestClient) -> None:
    """GET /health → 200 ve status=='healthy'."""
    response = client.get("/health")

    assert response.status_code == 200, f"/health 200 dönmeli, gelen {response.status_code}"
    body = response.json()
    assert body["status"] == "healthy", f"status 'healthy' olmalı, gelen {body.get('status')}"
