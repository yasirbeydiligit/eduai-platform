"""Soru endpoint'leri — sadece HTTP katmanı; iş mantığı QuestionService'de.

Router'ın sorumluluğu:
  1. İsteği deserialize etmek (Pydantic zaten yapar)
  2. Service'e delege etmek
  3. Yanıtı serialize etmek (FastAPI response_model zaten yapar)
  4. İlgili yan etkileri orkestre etmek (session activity update gibi)
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.dependencies import get_question_service, get_session_service
from app.schemas.questions import QuestionRequest, QuestionResponse
from app.services.question_service import QuestionService
from app.services.session_service import SessionService

router = APIRouter(prefix="/v1/questions", tags=["Questions"])


@router.post(
    "/ask",
    response_model=QuestionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Bir soru sor",
    description="Soruyu işler, mock cevap döndürür ve oturum aktivitesini kaydeder.",
)
async def ask_question(
    request: QuestionRequest,
    # Annotated[T, Depends(fn)] modern FastAPI DI pattern — ruff B008 uyumu sağlar
    # (function call'ı argument default'ta değil, type annotation içinde taşır).
    question_service: Annotated[QuestionService, Depends(get_question_service)],
    session_service: Annotated[SessionService, Depends(get_session_service)],
) -> QuestionResponse:
    """Soruyu işle, cevabı döndür ve oturum istatistiklerini güncelle."""
    # Asıl iş: cevap üretimi. Mock P1'de; P3'te RAG + LLM gelecek.
    response = await question_service.process_question(request)

    # Yan etki: oturum aktivite kaydı (best-effort). Session service bulunamayan
    # session_id için sessizce warning log'lar, exception fırlatmaz → bu çağrı
    # asla ana akışı bozmaz. Router layer burada service'leri orkestre eder;
    # question_service'in session_service'i bilmesi coupling yaratırdı.
    await session_service.record_question(request.session_id, request.subject)

    return response
