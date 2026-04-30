"""FastAPI uygulama giriş noktası.

Sorumluluklar:
  - App instance'ı + lifespan (startup/shutdown)
  - CORS middleware
  - Router'ları mount etme
  - /health monitoring endpoint'i
  - Global exception handler (unhandled error'lar için structured log + 500)

P3 Task 5 entegrasyonu (Sapma 26-31):
  - Lifespan startup'ta DocumentIndexer + LangGraph pipeline + retriever
    pre-warm → ilk request cold-start cezası yok (Sapma 21 fix).
  - .env cascade: agents/.env → repo_root/.env → ml/.env (pipeline.py
    pattern'iyle uyumlu) → ANTHROPIC_API_KEY otomatik bulunur.
"""

import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import structlog
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


def _load_env_cascade() -> None:
    """agents/pipeline ile aynı cascade — repo root + agents + ml.

    Settings init'inden ÖNCE çağrılmalı; aksi halde pydantic-settings
    sadece services/api/.env'i okur (ki yoktur), QDRANT_URL/ANTHROPIC_API_KEY
    boş kalır.
    """
    # services/api/app/main.py → parent[3] = repo root.
    repo_root = Path(__file__).resolve().parents[3]
    for path in [
        repo_root / "agents" / ".env",
        repo_root / ".env",
        repo_root / "ml" / ".env",
    ]:
        if path.exists():
            load_dotenv(path)


_load_env_cascade()

# Settings .env yüklendikten sonra import + init edilmeli.
from app.core.config import get_settings  # noqa: E402
from app.routers import documents, questions, sessions  # noqa: E402

# Settings app başlatılmadan önce tek kez okunur (lru_cache'li).
settings = get_settings()

# Modül-seviye logger — lifespan event'leri ve global exception handler için.
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App yaşam döngüsü — startup ve shutdown hook'ları.

    Startup: P3 RAG/agent altyapısını eager initialize eder → ilk request
    cold-start cezası yaşamaz (e5-large model load 30+ sn olabiliyor).
    Hata durumunda app yine başlar; ilgili endpoint runtime'da hata verir
    (graceful degradation).
    """
    logger.info("eduai_api_starting", version=settings.VERSION, debug=settings.DEBUG)

    # --- P3 startup: DocumentIndexer + pipeline + retriever pre-warm ---
    # Import lazy: P1-only çalıştırmalar (örn. unit test) bu satırlara
    # dokunmadan geçebilsin. Hata yakalama: agents/ kütüphaneleri yoksa
    # health endpoint'i yine ayakta kalır.
    try:
        from agents.graph.nodes import _get_retriever
        from agents.graph.pipeline import build_pipeline
        from agents.rag.indexer import DocumentIndexer

        # Indexer — embedder + Qdrant client; collection yoksa create eder.
        # Vector_size erişimi embedder'ı eagerly load eder (~30 sn cold).
        app.state.indexer = DocumentIndexer()
        _ = app.state.indexer.embedder.vector_size  # pre-warm trigger

        # LangGraph pipeline — compiled graph, request başına yeniden
        # build edilmesin (build cost ~ms ama state machine pure).
        app.state.pipeline = build_pipeline()

        # Retriever singleton'ını da pre-warm — nodes.py module-level
        # global'i set olur, ilk /ask/v2 request'i hızlı.
        _get_retriever()

        logger.info(
            "p3_runtime_ready",
            collection=app.state.indexer.collection_name,
            vector_size=app.state.indexer.embedder.vector_size,
            llm_backend=os.getenv("LLM_BACKEND", "anthropic"),
            anthropic_key_present=bool(os.getenv("ANTHROPIC_API_KEY")),
        )
    except Exception as exc:
        # Eager init başarısız (Qdrant down, deps eksik vs) — uyar ama app'i
        # düşürme. /v1/documents ve /v1/questions/ask/v2 endpoint'leri
        # runtime'da hata verir; /health ve mock /ask çalışır.
        logger.error(
            "p3_runtime_init_failed",
            exc_type=type(exc).__name__,
            exc_msg=str(exc),
        )
        app.state.indexer = None
        app.state.pipeline = None

    yield

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
