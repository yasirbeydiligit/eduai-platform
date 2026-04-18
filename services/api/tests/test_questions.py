"""POST /v1/questions/ask endpoint'i için validasyon + happy-path testleri.

Her test fonksiyonu sadece bir davranışı doğrular (tek sorumluluk prensibi).
"""

from __future__ import annotations

from uuid import UUID, uuid4

from fastapi.testclient import TestClient


def _base_payload() -> dict:
    """Geçerli bir soru payload'u şablonu — testler üzerine ufak değişikliklerle kullanır."""
    return {
        "question": "Osmanlı İmparatorluğu ne zaman kuruldu?",
        "session_id": str(uuid4()),
        "subject": "tarih",
        "grade_level": 9,
    }


def test_ask_question_valid(client: TestClient) -> None:
    """Geçerli payload → 201, question_id UUID formatında, answer string döner."""
    # _base_payload() her çağrıda yeni UUID üretir — request/response karşılaştırması için
    # aynı payload instance'ını yerel değişkene alıyoruz.
    payload = _base_payload()
    response = client.post("/v1/questions/ask", json=payload)

    assert response.status_code == 201, (
        f"Beklenen 201, gelen {response.status_code}: {response.text}"
    )
    body = response.json()

    # UUID parse edilebiliyorsa formatı doğru demektir
    UUID(body["question_id"])  # exception fırlatmazsa OK
    assert isinstance(body["answer"], str) and body["answer"], "answer non-empty string olmalı"
    assert body["session_id"] == payload["session_id"], (
        "response session_id, request ile aynı olmalı"
    )


def test_ask_question_too_short(client: TestClient) -> None:
    """4 karakterlik soru min_length=5 kuralını ihlal eder → 422."""
    payload = _base_payload()
    payload["question"] = "abcd"  # 4 karakter, min 5

    response = client.post("/v1/questions/ask", json=payload)
    assert response.status_code == 422, (
        f"4 karakterlik soru 422 dönmeli, gelen {response.status_code}"
    )


def test_ask_question_too_long(client: TestClient) -> None:
    """501 karakterlik soru max_length=500 kuralını ihlal eder → 422."""
    payload = _base_payload()
    payload["question"] = "a" * 501

    response = client.post("/v1/questions/ask", json=payload)
    assert response.status_code == 422, (
        f"501 karakterlik soru 422 dönmeli, gelen {response.status_code}"
    )


def test_ask_question_invalid_subject(client: TestClient) -> None:
    """Enum dışı subject → 422.

    NOT: Enum'da 'cografya' (ASCII g) var; test 'coğrafya' (Türkçe ğ) kullanıyor,
    bu string enum'a eşleşmez → 422 bekleniyor. Bu davranış kasıtlıdır.
    """
    payload = _base_payload()
    payload["subject"] = "coğrafya"  # enum'da yok (ğ vs g farkı)

    response = client.post("/v1/questions/ask", json=payload)
    assert response.status_code == 422, (
        f"Geçersiz subject 422 dönmeli, gelen {response.status_code}"
    )


def test_ask_question_invalid_grade(client: TestClient) -> None:
    """grade_level=13 le=12 kuralını ihlal eder → 422."""
    payload = _base_payload()
    payload["grade_level"] = 13

    response = client.post("/v1/questions/ask", json=payload)
    assert response.status_code == 422, (
        f"grade_level=13 için 422 dönmeli, gelen {response.status_code}"
    )


def test_ask_question_invalid_grade_zero(client: TestClient) -> None:
    """grade_level=0 ge=1 kuralını ihlal eder → 422."""
    payload = _base_payload()
    payload["grade_level"] = 0

    response = client.post("/v1/questions/ask", json=payload)
    assert response.status_code == 422, (
        f"grade_level=0 için 422 dönmeli, gelen {response.status_code}"
    )
