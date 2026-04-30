"""Soru endpoint'leri — sadece HTTP katmanı; iş mantığı QuestionService'de.

Router'ın sorumluluğu:
  1. İsteği deserialize etmek (Pydantic zaten yapar)
  2. Service'e delege etmek
  3. Yanıtı serialize etmek (FastAPI response_model zaten yapar)
  4. İlgili yan etkileri orkestre etmek (session activity update gibi)

P3 Task 5 — `/ask/v2` endpoint'i LangGraph pipeline ile gerçek RAG cevabı
üretir. `/ask` (v1) mock olarak kalır (geriye uyum + dev test).
"""

from __future__ import annotations

import time
from typing import Annotated
from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends, status

from app.dependencies import (
    get_agent_pipeline,
    get_question_service,
    get_session_service,
)
from app.schemas.questions import QuestionRequest, QuestionResponse
from app.services.question_service import QuestionService
from app.services.session_service import SessionService

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/v1/questions", tags=["Questions"])


@router.post(
    "/ask",
    response_model=QuestionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Bir soru sor (P1 mock)",
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


@router.post(
    "/ask/v2",
    response_model=QuestionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Bir soru sor (P3 LangGraph RAG)",
    description=(
        "Soruyu LangGraph pipeline'ından geçirir: retrieve (Qdrant) → generate "
        "(Anthropic claude-haiku-4-5) → validate (kalite + retry) → format. "
        "Cevap markdown formatlı, kaynak listesi ve confidence skoru ile."
    ),
)
async def ask_question_v2(
    request: QuestionRequest,
    # LangGraph compiled StateGraph; type hint dependencies.py'de runtime
    # resolution gerektirmesin diye gevşek (object).
    pipeline: Annotated[object, Depends(get_agent_pipeline)],
    session_service: Annotated[SessionService, Depends(get_session_service)],
) -> QuestionResponse:
    """LangGraph pipeline ile RAG-grounded cevap üret."""
    start = time.monotonic()

    # AgentState — LangGraph TypedDict; JSON serializable olması için
    # session_id UUID → str. subject SubjectEnum → str.value (RAG payload
    # filter string match yapıyor; enum string'e çevrilmek zorunda).
    initial_state = {
        "question": request.question,
        "subject": request.subject.value,
        "grade_level": request.grade_level,
        "session_id": str(request.session_id),
        "attempts": 0,
        "needs_retry": False,
    }

    # Pipeline çalıştır — async invoke (LangGraph node'ları async).
    result = await pipeline.ainvoke(initial_state)

    elapsed_ms = int((time.monotonic() - start) * 1000)

    # Session record (best-effort; ana akışı bozmaz).
    await session_service.record_question(request.session_id, request.subject)

    response = QuestionResponse(
        question_id=uuid4(),
        answer=result.get("answer", ""),
        confidence=float(result.get("confidence", 0.0)),
        sources=result.get("sources", []),
        processing_time_ms=elapsed_ms,
        session_id=request.session_id,
    )

    logger.info(
        "ask_v2_processed",
        question_id=str(response.question_id),
        session_id=str(request.session_id),
        subject=request.subject.value,
        attempts=result.get("attempts", 0),
        confidence=response.confidence,
        sources_count=len(response.sources),
        processing_time_ms=elapsed_ms,
    )
    return response
