"""Soru işleme iş mantığı — HTTP katmanından bağımsız.

P1'de mock cevap üretir. P3 fazında gerçek AI (RAG + LLM) entegrasyonu bu servise
enjekte edilecek; router layer'da hiçbir değişiklik gerekmeyecek (separation of concerns).
"""

from __future__ import annotations

import time
from uuid import UUID, uuid4

import structlog

from app.schemas.questions import QuestionRequest, QuestionResponse

# Modül seviyesinde logger — tüm method'lar aynı logger'ı paylaşır.
# __name__ sayesinde log kaydında kaynak modül görünür.
logger = structlog.get_logger(__name__)


class QuestionService:
    """Soru-cevap iş mantığını yöneten servis. In-memory store kullanır (P1 kapsamı)."""

    def __init__(self) -> None:
        # Proses restart'ında silinen in-memory store.
        # P2/P3'te kalıcı storage (Postgres) ile değiştirilecek — interface aynı kalacak.
        self._store: dict[UUID, QuestionResponse] = {}

    async def process_question(self, request: QuestionRequest) -> QuestionResponse:
        """İsteği işle, mock cevap üret, store'a kaydet ve response döndür."""
        # time.monotonic() wall-clock değil monotonik saat — sistem saati geriye alınsa bile
        # duration ölçümü güvenli. time.time() kullanmak production'da hatalıdır.
        start = time.monotonic()

        question_id = uuid4()

        # P1 mock cevap — SubjectEnum str enum, .value = string ham değeri ("matematik" vs)
        answer = (
            f"[EduAI Mock] '{request.subject.value}' konusunda sorunuz alındı. "
            "P3 fazında gerçek AI cevabı buraya gelecek."
        )

        # Saniye → milisaniye; schema int bekliyor, explicit cast
        processing_time_ms = int((time.monotonic() - start) * 1000)

        response = QuestionResponse(
            question_id=question_id,
            answer=answer,
            confidence=0.0,  # P1 mock — güven skoru anlamsız
            sources=[],  # P3'te RAG kaynakları eklenecek
            processing_time_ms=processing_time_ms,
            session_id=request.session_id,
        )

        # Belleğe yaz (P1 için yeterli persistence)
        self._store[question_id] = response

        # structlog kwargs → structured log field'ları (JSON output'ta ayrı key'ler olur).
        # UUID'ler str'e çevrilir ki JSON serializer'da sorun çıkmasın.
        logger.info(
            "question_processed",
            question_id=str(question_id),
            session_id=str(request.session_id),
            subject=request.subject.value,
            grade_level=request.grade_level,
            processing_time_ms=processing_time_ms,
        )

        return response

    async def get_question(self, question_id: UUID) -> QuestionResponse | None:
        """Verilen ID'ye karşılık gelen soruyu döndür; yoksa None."""
        return self._store.get(question_id)
