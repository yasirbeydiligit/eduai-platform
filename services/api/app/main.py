"""FastAPI uygulama giriş noktası.

Sorumluluklar:
  - App instance'ı + lifespan (startup/shutdown)
  - CORS middleware
  - Router'ları mount etme
  - /health monitoring endpoint'i
  - Global exception handler (unhandled error'lar için structured log + 500)
"""

from contextlib import asynccontextmanager
from datetime import UTC, datetime

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.routers import documents, questions, sessions

# Settings app başlatılmadan önce tek kez okunur (lru_cache'li).
settings = get_settings()

# Modül-seviye logger — lifespan event'leri ve global exception handler için.
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App yaşam döngüsü — startup ve shutdown hook'ları.

    P2+P3'te: DB connection pool, Qdrant client, MLflow tracking vs. burada init edilecek.
    """
    # --- startup ---
    logger.info("eduai_api_starting", version=settings.VERSION, debug=settings.DEBUG)
    yield
    # --- shutdown ---
    logger.info("eduai_api_shutting_down")


# Swagger UI (/docs) için title + description — recruiter/developer ilk buraya bakar.
app = FastAPI(
    title="EduAI API",
    description=(
        "Türkçe lise eğitimi için AI destekli soru-cevap platformunun backend API'si. "
        "P1 kapsamı: FastAPI iskeleti + mock cevaplar. "
        "Gerçek AI entegrasyonu P3 fazında (RAG + LangGraph) eklenecek."
    ),
    version=settings.VERSION,
    debug=settings.DEBUG,
    lifespan=lifespan,
)

# CORS — development için tüm origin'lere açık.
# Production'da env-driven whitelist'e kısıt (örn. settings.ALLOWED_ORIGINS).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Router'ları sırayla bağla — her biri kendi prefix/tag'iyle mount olur.
# Swagger UI'da tag'ler gruplandırma için kullanılır.
app.include_router(questions.router)
app.include_router(documents.router)
app.include_router(sessions.router)


# Global exception handler — yakalanmamış (HTTPException DIŞI) tüm hatalar buraya düşer.
# HTTPException zaten FastAPI tarafından işlenir (404/413/415/422 vs.); bu handler 500'e
# yönelik. Structured log alanları (path, method, exc_type, exc_msg) production aggregator'da
# (Sentry/Datadog/CloudWatch) filtreleme sağlar.
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Yakalanmamış exception'lar için fallback — structured log + 500 response."""
    logger.error(
        "unhandled_exception",
        path=request.url.path,
        method=request.method,
        exc_type=type(exc).__name__,
        exc_msg=str(exc),
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


@app.get("/health", tags=["Health"], summary="Sistem sağlık kontrolü")
async def health() -> dict:
    """Monitoring/load balancer için basit sağlık ucu.

    Not: datetime.UTC (Python 3.11+) timezone-aware UTC sabiti — utcnow() deprecated yerine.
    → JSON output ISO 8601 '+00:00' suffix'li olur, parsing/test assertion deterministik.
    """
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "timestamp": datetime.now(UTC).isoformat(),
        "services": {
            "api": "healthy",
            "storage": "healthy",
        },
    }
