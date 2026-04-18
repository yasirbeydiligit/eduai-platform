"""Oturum yönetimi için request/response şemaları."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.questions import SubjectEnum


# GET /v1/sessions/{session_id} dönüş modeli.
# Mevcut bir oturumun özet bilgilerini ve geçmiş aktivitesini döner.
class SessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    session_id: UUID = Field(
        ...,
        description="Oturumun benzersiz kimliği.",
    )
    created_at: datetime = Field(
        ...,
        description="Oturumun oluşturulma zamanı (UTC).",
    )
    question_count: int = Field(
        default=0,
        ge=0,
        description="Bu oturumda sorulmuş toplam soru sayısı.",
    )
    last_activity: datetime = Field(
        ...,
        description="Oturumdaki son etkileşim zamanı (UTC).",
    )
    # default_factory=list → her instance kendi mutable list'ine sahip olur.
    subjects_accessed: list[SubjectEnum] = Field(
        default_factory=list,
        description="Bu oturumda erişilen ders konularının listesi (tekrarsız beklenir).",
    )


# POST /v1/sessions/ dönüş modeli.
# Sadece yeni oluşturulan oturumun ID'sini ve zamanını döner — minimal payload.
class SessionCreateResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    session_id: UUID = Field(
        default_factory=uuid4,
        description="Yeni oluşturulan oturumun UUID'si.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Oturumun oluşturulma zamanı (UTC, timezone-aware).",
    )
