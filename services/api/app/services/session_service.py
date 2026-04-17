"""Oturum yönetimi — HTTP katmanından bağımsız.

Oturum (session), aynı kullanıcının art arda sorduğu soruları ilişkilendirmek için
kullanılır. P3'te agent'lar bu bağlamı kullanarak conversation history kuracak.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

import structlog

from app.schemas.questions import SubjectEnum
from app.schemas.sessions import SessionCreateResponse, SessionResponse

logger = structlog.get_logger(__name__)


class SessionService:
    """Oturum yaşam döngüsü ve aktivite takibi."""

    def __init__(self) -> None:
        # Internal şema — her UUID için şu dict tutulur:
        #   {
        #     "created_at":       datetime (UTC),
        #     "last_activity":    datetime (UTC),
        #     "question_count":   int,
        #     "subjects_accessed": set[SubjectEnum],   ← tekrarları otomatik önler
        #   }
        # Dışarı SessionResponse verirken set → sorted list'e çevrilir (deterministik output).
        self._store: dict[UUID, dict[str, Any]] = {}

    async def create_session(self) -> SessionCreateResponse:
        """Yeni boş oturum oluştur ve minimal yanıt döndür."""
        session_id = uuid4()
        # datetime.utcnow() deprecated — timezone-aware sürüm kullanıyoruz.
        # Store ile response'un aynı zaman değerini taşıması için now'ı yerel var'a alıyoruz.
        now = datetime.now(timezone.utc)

        self._store[session_id] = {
            "created_at": now,
            "last_activity": now,
            "question_count": 0,
            "subjects_accessed": set(),
        }

        logger.info("session_created", session_id=str(session_id))

        # Pydantic default_factory yeni UUID/timestamp üretirdi; store ile tutarlılık için
        # explicit geçiyoruz.
        return SessionCreateResponse(session_id=session_id, created_at=now)

    async def get_session(self, session_id: UUID) -> SessionResponse | None:
        """Oturum detaylarını getir; yoksa None."""
        data = self._store.get(session_id)
        if data is None:
            return None

        # set → sorted list. SubjectEnum str enum, .value alfabetik sırala (deterministik).
        return SessionResponse(
            session_id=session_id,
            created_at=data["created_at"],
            question_count=data["question_count"],
            last_activity=data["last_activity"],
            subjects_accessed=sorted(data["subjects_accessed"], key=lambda s: s.value),
        )

    async def record_question(self, session_id: UUID, subject: SubjectEnum) -> None:
        """Oturum istatistiklerini yeni bir soru için güncelle (sayı + son aktivite + konu)."""
        data = self._store.get(session_id)
        if data is None:
            # Oturum yoksa sessiz yutma — warning log bırak; router layer 404 döndürme
            # sorumluluğunda. Burada exception fırlatırsak router'ı kirletmiş oluruz.
            logger.warning(
                "record_question_unknown_session",
                session_id=str(session_id),
                subject=subject.value,
            )
            return

        data["question_count"] += 1
        data["last_activity"] = datetime.now(timezone.utc)
        # set.add idempotent — aynı subject tekrar eklense de liste büyümez
        data["subjects_accessed"].add(subject)

        logger.info(
            "session_question_recorded",
            session_id=str(session_id),
            subject=subject.value,
            question_count=data["question_count"],
        )
