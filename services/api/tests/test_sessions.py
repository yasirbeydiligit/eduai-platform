"""Session endpoint'leri — create ve read için testler.

NOT: Service singleton (lru_cache) olduğu için testler arası session store paylaşılır.
Bu testler birbirini etkilemeyecek şekilde yazılmıştır:
  - test_create_session: state create eder, kendi response'una bakar
  - test_get_session_exists: kendi session'ını oluşturur, onu getirir
  - test_get_session_not_found: fresh UUID4 kullanır (çakışma ihmal edilebilir)
"""

from __future__ import annotations

from uuid import UUID, uuid4

from fastapi.testclient import TestClient


def test_create_session(client: TestClient) -> None:
    """POST /v1/sessions/ → 201 + geçerli UUID session_id."""
    response = client.post("/v1/sessions/")

    assert response.status_code == 201, (
        f"Beklenen 201, gelen {response.status_code}: {response.text}"
    )
    body = response.json()

    UUID(body["session_id"])  # UUID parse edilmezse AssertionError yerine ValueError → test fail
    assert "created_at" in body, "created_at alanı response'da olmalı"


def test_get_session_exists(client: TestClient) -> None:
    """Önce oturum oluştur, sonra GET → 200 ve aynı session_id dönmeli."""
    # 1) Session oluştur
    create_resp = client.post("/v1/sessions/")
    assert create_resp.status_code == 201, "Setup: session oluşturulamadı"
    session_id = create_resp.json()["session_id"]

    # 2) Aynı session_id ile GET
    get_resp = client.get(f"/v1/sessions/{session_id}")

    assert get_resp.status_code == 200, (
        f"Var olan session için 200 dönmeli, gelen {get_resp.status_code}"
    )
    body = get_resp.json()
    assert body["session_id"] == session_id, "Dönen session_id, create'teki ile aynı olmalı"
    assert body["question_count"] == 0, "Yeni session için question_count=0 olmalı"


def test_get_session_not_found(client: TestClient) -> None:
    """Hiç oluşturulmamış rastgele UUID → 404."""
    # Fresh UUID4 — mevcut store'la çakışma olasılığı pratik olarak sıfır (2^122 entropi)
    random_id = uuid4()

    response = client.get(f"/v1/sessions/{random_id}")

    assert response.status_code == 404, (
        f"Bulunmayan session için 404 dönmeli, gelen {response.status_code}"
    )
