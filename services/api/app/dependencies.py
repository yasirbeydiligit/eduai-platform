"""FastAPI dependency injection factory'leri.

Service instance'ları app yaşam süresi boyunca **tek** kez oluşturulur (singleton).
Bunun sebebi: service'ler in-memory store tutuyor — her request'te yeni instance
yaratsak store sıfırlanırdı. @lru_cache singleton pattern'i FastAPI'nin resmi
önerisidir (module-level global'den daha test-friendly: override edilebilir, clear'lanabilir).

P3 Task 5: agent pipeline ve document indexer için ayrı pattern — bunlar
lifespan'de eager init'leniyor ve `app.state`'e yazılıyor (ağır objeler;
@lru_cache pattern'i request bağlamında app'e erişemez). `Request` injection
ile state'e erişiyoruz.
"""

from functools import lru_cache

# agents/ runtime import: FastAPI Depends() introspection'ı tip nesnesinin
# gerçekten yüklü olmasını gerektiriyor (TYPE_CHECKING bloğunda kalırsa
# parametre query param olarak yorumlanır → 422). P1-only test çalıştırmak
# isteyenler için: agents/ tüm dependency'leri venv'de yüklü olmalı.
from agents.rag.indexer import DocumentIndexer
from fastapi import HTTPException, Request, status

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


# --- P3 Task 5: agents/ entegrasyonu ---


def get_document_indexer(request: Request) -> DocumentIndexer:
    """Lifespan'de oluşturulan DocumentIndexer'ı app.state'ten döndür.

    Indexer None ise (startup'ta init başarısız oldu, Qdrant down vs.)
    503 fırlat — endpoint kullanıcısına altyapı sorunu olduğunu söyle.
    """
    indexer = getattr(request.app.state, "indexer", None)
    if indexer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Document indexer hazır değil — Qdrant veya embedder "
                "altyapısı kontrol edilmeli."
            ),
        )
    return indexer


def get_agent_pipeline(request: Request):
    """Lifespan'de compile edilen LangGraph pipeline'ı döndür.

    Compiled StateGraph type'ını runtime'da resolve etmemek için annotation yok
    (LangGraph internal class).
    """
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent pipeline hazır değil — startup logs kontrol edilmeli.",
        )
    return pipeline
