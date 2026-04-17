"""Döküman metadata yönetimi — HTTP katmanından bağımsız.

P1'de sadece metadata in-memory tutulur; dosya içeriği kaydedilmez veya işlenmez.
P3'te Qdrant'a vektörleme pipeline'ı bu servise eklenecek ve status alanı
"uploaded" → "processing" → "ready" akışını izleyecek.
"""

from __future__ import annotations

from uuid import UUID

import structlog

from app.schemas.documents import DocumentUploadResponse
from app.schemas.questions import SubjectEnum

logger = structlog.get_logger(__name__)


class DocumentService:
    """Döküman metadata iş mantığını yöneten servis."""

    def __init__(self) -> None:
        # Sadece metadata — dosya içeriği/binary burada değil.
        # P3'te gerçek dosya storage (S3/local volume) eklenecek.
        self._store: dict[UUID, DocumentUploadResponse] = {}

    async def save_document(
        self,
        title: str,
        subject: SubjectEnum,
        grade_level: int,
        file_size: int,
    ) -> DocumentUploadResponse:
        """Döküman metadata'sını mock olarak kaydet ve response döndür."""
        # Pydantic modeli document_id (uuid4) ve created_at'i (utc now) default_factory
        # ile otomatik üretir; burada tekrar etmiyoruz → single source of truth schema'da.
        document = DocumentUploadResponse(
            title=title,
            subject=subject,
            grade_level=grade_level,
            file_size_bytes=file_size,
            status="uploaded",  # P3'te async processing tamamlanınca "ready"e güncellenir
        )

        self._store[document.document_id] = document

        logger.info(
            "document_saved",
            document_id=str(document.document_id),
            title=title,
            subject=subject.value,
            grade_level=grade_level,
            file_size_bytes=file_size,
        )

        return document

    async def get_document(self, document_id: UUID) -> DocumentUploadResponse | None:
        """Verilen ID'ye karşılık gelen dökümanı döndür; yoksa None."""
        return self._store.get(document_id)
