"""Oturum endpoint'leri — minimal CRUD (create + read)."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import get_session_service
from app.schemas.sessions import SessionCreateResponse, SessionResponse
from app.services.session_service import SessionService

router = APIRouter(prefix="/v1/sessions", tags=["Sessions"])


@router.post(
    "/",
    response_model=SessionCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Yeni oturum oluştur",
    description="Boş bir oturum oluşturur ve yeni oturum ID'sini döndürür.",
)
async def create_session(
    service: SessionService = Depends(get_session_service),
) -> SessionCreateResponse:
    """Yeni boş oturum oluştur, UUID'sini döndür."""
    return await service.create_session()


@router.get(
    "/{session_id}",
    response_model=SessionResponse,
    summary="Oturum detaylarını getir",
    description="Oturum özeti: oluşturulma zamanı, soru sayısı, erişilen konular.",
    responses={404: {"description": "Session not found"}},
)
async def get_session(
    # FastAPI UUID path param'ı otomatik parse eder; geçersiz format → 422
    session_id: UUID,
    service: SessionService = Depends(get_session_service),
) -> SessionResponse:
    """Verilen ID ile oturumu getir; bulunamazsa 404."""
    session = await service.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )
    return session
