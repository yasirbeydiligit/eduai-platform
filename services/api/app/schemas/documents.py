"""Döküman yükleme akışı için response şemaları."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

# SubjectEnum'u tekrar tanımlamıyoruz — tek doğruluk kaynağı questions modülü.
from app.schemas.questions import SubjectEnum


# POST /v1/documents/upload endpoint'inin dönüş modeli.
# Not: P1'de yüklenen dosya in-memory metadata olarak tutulur; gerçek içerik işlenmez.
# P3'te Qdrant'a vektörize edilerek RAG pipeline'ına eklenecek.
class DocumentUploadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    document_id: UUID = Field(
        default_factory=uuid4,
        description="Yüklenen dökümanın benzersiz kimliği.",
    )
    title: str = Field(
        ...,
        max_length=200,
        description="Döküman başlığı (max 200 karakter).",
    )
    subject: SubjectEnum = Field(
        ...,
        description="Dökümanın ait olduğu ders konusu.",
    )
    # SPEC'te açıkça belirtilmemiş olsa da grade_level her yerde 1-12 bound'u taşıyor;
    # tutarlılık için burada da uyguluyoruz.
    grade_level: int = Field(
        ...,
        ge=1,
        le=12,
        description="Dökümanın hedef sınıf seviyesi (1-12).",
    )
    file_size_bytes: int = Field(
        ...,
        ge=0,
        description="Dosya boyutu (byte).",
    )
    # Durum akışı: uploaded → processing (vektörleme) → ready (sorgulanabilir).
    # Literal kullanımı enum'a göre daha hafif: küçük ve sabit değer kümeleri için ideal.
    status: Literal["uploaded", "processing", "ready"] = Field(
        ...,
        description="Dökümanın işlenme aşaması.",
    )
    # P3 Task 5 ek alan — Qdrant'a yüklenen chunk sayısı.
    # 0 → henüz indekslenmedi veya zaten indeksli (duplicate-skip).
    # Sapma 28: SPEC'in DocumentUploadResponse'unda yoktu; TASKS.md ekstra
    # gereksinimi.
    chunks_indexed: int = Field(
        default=0,
        ge=0,
        description="Qdrant'a yüklenen chunk sayısı (0 = duplicate-skip veya henüz indekslenmedi).",
    )
    # datetime.UTC (Python 3.11+) timezone-aware UTC sabiti — utcnow() deprecated yerine.
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Yükleme zamanı (UTC, timezone-aware).",
    )
