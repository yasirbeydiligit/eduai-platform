"""Soru-cevap akışı için request/response şemaları."""

from __future__ import annotations

from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


# Desteklenen ders konuları — platformun tamamında ortak enum olarak kullanılır.
# str miras alındığı için JSON'da doğrudan string olarak serileşir ve Swagger'da
# dropdown olarak görünür.
class SubjectEnum(str, Enum):
    MATEMATIK = "matematik"
    FIZIK = "fizik"
    KIMYA = "kimya"
    BIYOLOJI = "biyoloji"
    TARIH = "tarih"
    COGRAFYA = "cografya"
    FELSEFE = "felsefe"
    DIN = "din"
    EDEBIYAT = "edebiyat"
    INGILIZCE = "ingilizce"
    GENEL = "genel"


# POST /v1/questions/ask endpoint'ine gelen istek gövdesi.
class QuestionRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    question: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Kullanıcının sorduğu soru metni (5-500 karakter).",
    )
    session_id: UUID = Field(
        ...,
        description="Bu sorunun ait olduğu oturumun UUID'si. Oturumlar arası bağlamı korur.",
    )
    subject: SubjectEnum = Field(
        ...,
        description="Sorunun ait olduğu ders konusu.",
    )
    grade_level: int = Field(
        ...,
        ge=1,
        le=12,
        description="Hedef sınıf seviyesi (Türk eğitim sistemi: 1-12).",
    )


# POST /v1/questions/ask endpoint'inin dönüş modeli.
# P1'de mock cevap döner; P3'te gerçek AI cevabı ile değiştirilecek.
class QuestionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    question_id: UUID = Field(
        default_factory=uuid4,
        description="Soruya atanan benzersiz kimlik.",
    )
    answer: str = Field(
        ...,
        description="AI modelinin (veya P1'de mock servisin) ürettiği cevap.",
    )
    # 0.0 = güven yok, 1.0 = tam güven. P1 mock değer döner (0.0).
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Cevabın güven skoru (0.0-1.0).",
    )
    # Mutable default kullanmamak için default_factory. list() her instance'a yeni liste verir.
    sources: list[str] = Field(
        default_factory=list,
        description="Cevabın dayandığı kaynak listesi (URL veya document_id).",
    )
    processing_time_ms: int = Field(
        ...,
        ge=0,
        description="İşlem süresi (milisaniye).",
    )
    session_id: UUID = Field(
        ...,
        description="İsteğin geldiği oturumun UUID'si (request ile aynı).",
    )
