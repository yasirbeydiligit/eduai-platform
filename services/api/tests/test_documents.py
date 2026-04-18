"""POST /v1/documents/upload endpoint'i için upload + validasyon testleri.

Multipart form-data ile dosya + metadata gönderimi test edilir.
io.BytesIO kullanarak fiziksel dosya yaratmadan in-memory dosya simüle ediyoruz.
"""

from __future__ import annotations

import io
from uuid import UUID

from fastapi.testclient import TestClient

# Test constants — magic number'lardan kaçınmak için named.
ONE_MB = 1024 * 1024
ELEVEN_MB = 11 * ONE_MB


def _form_data() -> dict:
    """Standart metadata — testlerde sadece dosya değişir."""
    return {
        "title": "Test Dökümanı",
        "subject": "matematik",
        "grade_level": 9,
    }


def test_upload_valid_txt(client: TestClient) -> None:
    """Geçerli .txt dosyası → 201, document_id UUID döner."""
    # BytesIO: dosyayı RAM'de tut, diske yazma. Multipart gönderim için yeterli.
    file_content = b"Bu bir test icerigidir."
    files = {"file": ("ornek.txt", io.BytesIO(file_content), "text/plain")}

    response = client.post("/v1/documents/upload", files=files, data=_form_data())

    assert response.status_code == 201, f"Geçerli upload 201 dönmeli, gelen {response.status_code}: {response.text}"
    body = response.json()

    # document_id UUID formatında olmalı — UUID(str) exception fırlatmazsa format doğru
    UUID(body["document_id"])
    assert body["title"] == "Test Dökümanı"
    assert body["file_size_bytes"] == len(file_content), "file_size_bytes yüklenen içeriğin byte sayısına eşit olmalı"
    assert body["status"] == "uploaded"


def test_upload_invalid_format(client: TestClient) -> None:
    """.exe dosyası ALLOWED_EXTENSIONS dışı → 415."""
    files = {"file": ("malware.exe", io.BytesIO(b"sahte exe"), "application/octet-stream")}

    response = client.post("/v1/documents/upload", files=files, data=_form_data())

    assert response.status_code == 415, (
        f".exe dosyası için 415 dönmeli, gelen {response.status_code}"
    )


def test_upload_too_large(client: TestClient) -> None:
    """11MB dosya 10MB sınırını aşar → 413."""
    # b"0" * N ile 11MB buffer yarat — bellek kullanımı CI için tolere edilebilir
    large_content = b"0" * ELEVEN_MB
    files = {"file": ("big.txt", io.BytesIO(large_content), "text/plain")}

    response = client.post("/v1/documents/upload", files=files, data=_form_data())

    assert response.status_code == 413, (
        f"11MB dosya için 413 dönmeli, gelen {response.status_code}"
    )
