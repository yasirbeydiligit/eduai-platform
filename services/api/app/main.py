"""FastAPI uygulama giriş noktası.

Sorumluluklar:
  - App instance'ı + lifespan (startup/shutdown)
  - CORS middleware
  - Router'ları mount etme
  - /health monitoring endpoint'i
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.routers import documents, questions, sessions

# Settings app başlatılmadan önce tek kez okunur (lru_cache'li).
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App yaşam döngüsü — startup ve shutdown hook'ları.

    P2+P3'te: DB connection pool, Qdrant client, MLflow tracking vs. burada init edilecek.
    """
    # --- startup ---
    yield
    # --- shutdown ---


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


@app.get("/health", tags=["Health"], summary="Sistem sağlık kontrolü")
async def health() -> dict:
    """Monitoring/load balancer için basit sağlık ucu.

    Not: datetime.utcnow() Python 3.12+ deprecated. timezone-aware sürüm kullanıyoruz
    → JSON output ISO 8601 '+00:00' suffix'li olur, parsing daha güvenli.
    """
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {
            "api": "healthy",
            "storage": "healthy",
        },
    }
