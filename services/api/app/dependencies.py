"""FastAPI dependency injection factory'leri.

Service instance'ları app yaşam süresi boyunca **tek** kez oluşturulur (singleton).
Bunun sebebi: service'ler in-memory store tutuyor — her request'te yeni instance
yaratsak store sıfırlanırdı. @lru_cache singleton pattern'i FastAPI'nin resmi
önerisidir (module-level global'den daha test-friendly: override edilebilir, clear'lanabilir).
"""

from __future__ import annotations

from functools import lru_cache

from app.core.config import Settings, get_settings
from app.services.document_service import DocumentService
from app.services.question_service import QuestionService
from app.services.session_service import SessionService


def settings_dependency() -> Settings:
    """Settings singleton'ı döndür (get_settings zaten lru_cache'li)."""
    return get_settings()


# NOT: lru_cache argümansız fonksiyonları ilk çağrıda execute eder, sonrakileri cache'den
# döner. Test'lerde farklı instance gerekirse `get_question_service.cache_clear()` ile reset.
@lru_cache
def get_question_service() -> QuestionService:
    """Uygulama ömrü boyunca tek QuestionService instance'ı döndür."""
    return QuestionService()


@lru_cache
def get_document_service() -> DocumentService:
    """Uygulama ömrü boyunca tek DocumentService instance'ı döndür."""
    return DocumentService()


@lru_cache
def get_session_service() -> SessionService:
    """Uygulama ömrü boyunca tek SessionService instance'ı döndür."""
    return SessionService()
