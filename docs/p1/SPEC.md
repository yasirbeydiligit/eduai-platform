# P1 — EduAI API: Proje Spesifikasyonu

> **P1 Status:** Tamamlandı — 2026-04-18. Kontrol listesi dosyanın sonunda.
>
> Bu SPEC P1 implementation'ının **gerçek sonuç durumunu** yansıtır. Orijinal spec git history'de (`git log docs/p1/SPEC.md`). Her bilinçli sapma için inline **📝 Implementation note** callout'u eklenmiş; tam karar tarihçesi `docs/p1/IMPLEMENTATION_NOTES_ARCHIVED.md`'dedir.
>
> P2'ye geçerken bu dosya + `services/api/` kodu + `README.md` fresh Claude session'a verilebilir.

---

## Proje özeti

**Ne:** Türkçe eğitim içerikleri için AI-powered soru-cevap platformunun backend API'si.
**Amaç:** Sonraki fazlarda (fine-tuning, agent sistemi, cloud deploy) kullanılacak sağlam temel.
**Şu anki kapsam:** FastAPI + Docker + GitHub Actions. AI yok henüz — sadece mimari + mock cevaplar.

---

## Teknik gereksinimler

### Stack
- Python 3.11+
- FastAPI 0.111+
- Pydantic v2
- Uvicorn (ASGI server, `[standard]` extras)
- pytest + httpx (testler için)
- structlog (structured logging)
- ruff (linter + formatter)
- Docker + Docker Compose
- GitHub Actions

### Klasör yapısı

```
eduai-platform/
├── services/
│   └── api/
│       ├── app/
│       │   ├── __init__.py
│       │   ├── main.py                          # FastAPI app + lifespan + CORS + /health + global exc handler
│       │   ├── routers/
│       │   │   ├── __init__.py
│       │   │   ├── questions.py
│       │   │   ├── documents.py
│       │   │   └── sessions.py
│       │   ├── schemas/
│       │   │   ├── __init__.py
│       │   │   ├── questions.py                 # SubjectEnum(StrEnum), QuestionRequest, QuestionResponse
│       │   │   ├── documents.py
│       │   │   └── sessions.py
│       │   ├── services/
│       │   │   ├── __init__.py
│       │   │   ├── question_service.py          # In-memory store, mock cevap, time.monotonic timing
│       │   │   ├── document_service.py
│       │   │   └── session_service.py           # set-based subjects_accessed, warning-on-missing
│       │   ├── core/
│       │   │   ├── __init__.py
│       │   │   └── config.py                    # Pydantic Settings + @lru_cache
│       │   └── dependencies.py                  # @lru_cache singleton factories
│       ├── tests/
│       │   ├── __init__.py
│       │   ├── conftest.py                      # TestClient fixture
│       │   ├── test_questions.py
│       │   ├── test_documents.py
│       │   ├── test_sessions.py
│       │   └── test_health.py
│       ├── Dockerfile                           # Multi-stage, non-root user, curl HEALTHCHECK
│       ├── requirements.txt                     # Production deps only (5 paket)
│       ├── requirements-dev.txt                 # + httpx, pytest, pytest-asyncio, ruff
│       ├── .dockerignore
│       └── pyproject.toml                       # ruff + pytest config
├── docs/
│   └── p1/
│       ├── CONCEPT.md
│       ├── SPEC.md                              # (this file)
│       ├── TASKS.md
│       ├── IMPLEMENTATION_NOTES_ARCHIVED.md     # P1 karar tarihçesi (archived)
│       └── P2_HANDOFF.md                        # Fresh Claude session P2'ye başlarken bağlam
├── docker-compose.yml                           # Dev: --reload + healthcheck + bind mount
├── .github/workflows/ci.yml                     # 3 job: lint → test → docker-build
├── .gitignore
└── README.md
```

> 📝 **Implementation notes — klasör yapısı:**
> - `core/logging.py` orijinal spec'te vardı, implement edilmedi: structlog config app-wide yapıldığı için ayrı modül gerek duyulmadı.
> - `requirements-dev.txt` eklendi — dev deps (pytest, httpx, ruff) production imajından uzak tutuldu (image ~340MB → ~284MB).
> - `.dockerignore` eklendi — `tests/`, `.git`, `__pycache__` imajdan hariç.
> - Test dosyaları genişletildi: `test_documents.py`, `test_sessions.py`, `test_health.py` eklendi (SPEC sadece `test_questions.py` istiyordu).
> - `IMPLEMENTATION_NOTES_ARCHIVED.md` + `P2_HANDOFF.md` eklendi (P1 kapama artifact'leri).

---

## Endpoint spesifikasyonları

### POST /v1/questions/ask

**Status code:** `201 Created`

**Request:**
```json
{
  "question": "Türkiye'de Osmanlı İmparatorluğu ne zaman kuruldu?",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "subject": "tarih",
  "grade_level": 9
}
```

**Response:**
```json
{
  "question_id": "uuid",
  "answer": "[EduAI Mock] 'tarih' konusunda sorunuz alındı. P3 fazında gerçek AI cevabı buraya gelecek.",
  "confidence": 0.0,
  "sources": [],
  "processing_time_ms": 0,
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Validasyon kuralları:**
- `question`: min 5, max 500 karakter
- `session_id`: geçerli UUID4
- `subject`: `SubjectEnum` (StrEnum) — değerler: `matematik, fizik, kimya, biyoloji, tarih, cografya, felsefe, din, edebiyat, ingilizce, genel`
- `grade_level`: int, 1-12 arası

> 📝 **Implementation notes:**
> - **SubjectEnum genişletildi** (kullanıcı tarafından): 8 değer → 11 değer (`cografya`, `felsefe`, `din` eklendi).
> - `SubjectEnum` **`StrEnum` (Python 3.11+)** olarak tanımlandı — `(str, Enum)` çoklu miras yerine standart tekil miras.
> - Response `sources: list[str]` tipiyle explicit tanımlandı (orijinal spec tipi belirtmemişti).
> - `processing_time_ms` için `ge=0` bound eklendi (defensive).
> - Router `POST /ask` endpoint'i ayrıca `session_service.record_question(...)` çağırır → oturum aktivitesi güncellenir. Bu, HTTP kontratı dışında bir yan etki; router = orchestration layer kararı.

---

### POST /v1/documents/upload

**Status code:** `201 Created`

**Request:** `multipart/form-data`
- `file`: PDF veya TXT, max 10MB
- `subject`: string (SubjectEnum değerleri)
- `title`: string, max 200 karakter
- `grade_level`: int, 1-12

**Error responses:**
- `413 Request Entity Too Large` — File too large (>10MB)
- `415 Unsupported Media Type` — Only PDF and TXT files allowed
- `422 Unprocessable Entity` — Pydantic validation error

**Response:**
```json
{
  "document_id": "uuid",
  "title": "...",
  "subject": "matematik",
  "grade_level": 9,
  "file_size_bytes": 102400,
  "status": "uploaded",
  "created_at": "2026-04-13T10:00:00+00:00"
}
```

**Not:** Şu an dosyayı sadece metadata olarak kaydet (in-memory). P3'te Qdrant'a vektörize edilecek.

> 📝 **Implementation notes:**
> - `grade_level` için `ge=1, le=12` bound eklendi (diğer şemalarla tutarlılık).
> - `UploadFile.size` None ise fallback: içerik okunup length ölçülür, `seek(0)` ile pointer reset (bazı client'lar content-length header göndermez).
> - `created_at` timezone-aware ISO 8601 (`+00:00` suffix'li) — `datetime.utcnow()` deprecated yerine `datetime.now(UTC)`.

---

### GET /v1/sessions/{session_id}

**Status codes:** `200 OK` | `404 Not Found`

**Response:**
```json
{
  "session_id": "uuid",
  "created_at": "2026-04-13T10:00:00+00:00",
  "question_count": 5,
  "last_activity": "2026-04-13T10:30:00+00:00",
  "subjects_accessed": ["matematik", "tarih"]
}
```

### POST /v1/sessions/

**Status code:** `201 Created`

**Response:**
```json
{
  "session_id": "uuid",
  "created_at": "2026-04-13T10:00:00+00:00"
}
```

> 📝 **Implementation notes:**
> - Internal `subjects_accessed` **`set[SubjectEnum]`** olarak tutulur (aynı konu tekrar eklense liste şişmez), response'ta `sorted(list, key=s.value)` ile deterministik liste'ye çevrilir.
> - `record_question(session_id, subject)` bilinmeyen session için sessiz `logger.warning` + return (exception fırlatmaz — 404 sorumluluğu router'da).

---

### GET /health

**Status code:** `200 OK`

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": "2026-04-13T10:00:00+00:00",
  "services": {
    "api": "healthy",
    "storage": "healthy"
  }
}
```

> 📝 **Implementation note:** `timestamp` `datetime.now(UTC).isoformat()` → ISO 8601 `+00:00` suffix'li string.

---

## Mimari kurallar — bunlara kesinlikle uy

### 1. Router sadece HTTP katmanı

```python
# YANLIŞ — iş mantığı router'da:
@router.post("/ask")
async def ask_question(request: QuestionRequest):
    answer = f"Mock: {request.question}"
    return QuestionResponse(answer=answer)

# DOĞRU — P1 final pattern (Annotated):
from typing import Annotated
from fastapi import Depends

@router.post("/ask")
async def ask_question(
    request: QuestionRequest,
    service: Annotated[QuestionService, Depends(get_question_service)],
):
    return await service.process_question(request)
```

> 📝 **Implementation note:** `= Depends(fn)` argument default syntax yerine **`Annotated[T, Depends(fn)]`** FastAPI 0.95+ pattern'i kullanıldı (ruff B008 uyumu + FastAPI resmi olarak önerdiği modern syntax). P2/P3'te yeni router/endpoint yazarken bu pattern kullanılmalı.

### 2. Settings — environment variable'lardan oku

```python
# core/config.py
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME: str = "EduAI API"
    VERSION: str = "0.1.0"
    DEBUG: bool = False
    MAX_UPLOAD_SIZE_MB: int = 10

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()
```

> 📝 **Implementation note:** Pydantic v2 `model_config = SettingsConfigDict(...)` kullanıldı (v1'deki `class Config:` yerine). `get_settings()` `@lru_cache`'li singleton — FastAPI DI'da override edilebilir.

### 3. Logging — print() yok, structlog var

```python
import structlog
logger = structlog.get_logger(__name__)
logger.info("question_processed", question_id=str(question_id), duration_ms=42)
```

> 📝 **Implementation note:** UUID değerleri log field'larında `str(...)` ile geçirilir — structlog JSON serializer tuzaklarından kaçınmak için.

### 4. Error handling — global exception handler

```python
# main.py — unhandled exception'lar için global handler (HTTPException DIŞI).
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "unhandled_exception",
        path=request.url.path,
        method=request.method,
        exc_type=type(exc).__name__,
        exc_msg=str(exc),
    )
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
```

> 📝 **Implementation note:** HTTPException (404/413/415/422) FastAPI tarafından işlenir; global handler sadece **unhandled** exception'lar için 500 + structured log.

### 5. Lifespan — startup/shutdown için @asynccontextmanager

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    logger.info("eduai_api_starting", version=settings.VERSION)
    yield
    # shutdown
    logger.info("eduai_api_shutting_down")

app = FastAPI(lifespan=lifespan)
```

### 6. Singleton services — `@lru_cache`

```python
# dependencies.py
from functools import lru_cache

@lru_cache
def get_question_service() -> QuestionService:
    return QuestionService()
```

> 📝 **Implementation note:** Module-level global yerine `@lru_cache` kullanıldı — FastAPI resmi önerisi; test'lerde `.cache_clear()` ile reset edilebilir.

---

## Dockerfile — multi-stage + non-root + healthcheck

```dockerfile
# syntax=docker/dockerfile:1.6

# Stage 1 — builder: user-site'a kur, stage 2'ye kopyalanabilir
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2 — production
FROM python:3.11-slim

# curl: HEALTHCHECK için (python:slim'de default yok)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user — principle of least privilege. UID 1000 host uyumu.
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Builder'dan user-site paketleri kopyala (appuser ownership).
# NOT: /root/.local → /home/appuser/.local (appuser /root/ 0700'e erişemez).
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local
COPY --chown=appuser:appuser app/ app/

ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

**Image boyutu:** ~284 MB. SPEC `<200MB` hedefi karşılanmadı — bilinçli kabul (Alpine + pydantic v2 Rust wheel uyumsuzluğu, P2'de PyTorch için de problematik). P4 (K8s) geçişinde distroless/multi-stage scratch yeniden değerlendirilecek.

---

## docker-compose.yml

```yaml
version: "3.9"

services:
  api:
    build: ./services/api
    ports:
      - "8000:8000"
    environment:
      - DEBUG=true
      - APP_NAME=EduAI API Dev
    volumes:
      - ./services/api/app:/app/app  # hot reload
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

Dev `command:` `--reload` (workers=1 force), production Dockerfile CMD `--workers 2`.

---

## GitHub Actions — ci.yml (3 job pipeline)

```yaml
name: "CI — EduAI API"

on: [push, pull_request]

jobs:
  lint:
    name: "Code Quality"
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"
      - run: pip install ruff
      - run: ruff check services/api/ --output-format=github
      - run: ruff format services/api/ --check

  test:
    name: "Tests"
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"
      - run: pip install -r services/api/requirements-dev.txt
      - working-directory: services/api
        run: pytest tests/ -v --tb=short

  docker-build:
    name: "Docker Build"
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - run: docker build -t eduai-api:ci services/api/
      - run: docker images eduai-api:ci
```

> 📝 **Implementation notes:**
> - **3 ayrı job** (lint/test/docker-build) — `needs:` ile fail-fast chain, SPEC'teki 2 job yerine.
> - `cache: "pip"` — ikinci run'dan itibaren install saniyelere iner.
> - Test job `requirements-dev.txt` kullanır (Task 5 dev/prod ayrımı sonrası).
> - Docker size check'te `--format '{{.Size}}'` **kaldırıldı** — GitHub Actions `{{ }}` template'ını expression olarak yorumluyor, `Invalid workflow file` hatası veriyordu. Plain `docker images` zaten SIZE sütununu içerir.

---

## Test gereksinimleri

Toplam **13 test** yazıldı (4 dosya):

```python
# test_questions.py — 6 test
test_ask_question_valid                     # 201 + UUID + answer
test_ask_question_too_short                 # 4 karakter → 422
test_ask_question_too_long                  # 501 karakter → 422
test_ask_question_invalid_subject           # enum dışı → 422
test_ask_question_invalid_grade             # grade_level=13 → 422
test_ask_question_invalid_grade_zero        # grade_level=0 → 422

# test_documents.py — 3 test
test_upload_valid_txt                       # valid .txt → 201
test_upload_invalid_format                  # .exe → 415
test_upload_too_large                       # 11MB → 413

# test_sessions.py — 3 test
test_create_session                         # POST → 201 + UUID
test_get_session_exists                     # create + GET → 200
test_get_session_not_found                  # random UUID → 404

# test_health.py — 1 test
test_health_ok                              # GET /health → 200
```

**Test pattern'leri:**
- `TestClient(app)` context manager (`conftest.py`'de fixture)
- Multipart upload testlerinde `io.BytesIO`
- UUID doğrulama: `UUID(str_value)` exception fırlatmazsa format doğru

---

## README.md gereksinimleri

README şunları içerir (**İngilizce**, recruiter/senior dev ready):
- CI + Python badge'leri
- Projenin ne olduğu (3 cümle)
- Tech stack tablosu ("neden" kolonu ile)
- Quick Start (`docker-compose up`)
- Endpoint tablosu + her biri için curl örneği
- Proje yapısı açıklaması
- 4 fazlık proje bağlamı + Learning Goals

---

## Teslim kriterleri — P1 Final

- [x] `docker-compose up` komutu çalışıyor, `localhost:8000/docs` açılıyor
- [x] Tüm core endpoint'ler `/docs`'ta görünüyor (5 endpoint + `/docs` + `/openapi.json`)
- [x] `pytest` geçiyor (**13 test**, SPEC min 4)
- [x] `ruff check` sıfır hata
- [x] `ruff format --check` temiz
- [x] GitHub Actions **3 job yeşil** (lint / test / docker-build)
- [ ] Docker image <200MB — **284MB, bilinçli kabul** (Alpine uyumsuzluğu)
- [x] README İngilizce, curl örnekleri dahil

---

## P1 → P2 hand-off

P2 fresh session için kickoff paketi: `docs/p1/P2_HANDOFF.md`.

P2 kapsamı: PyTorch + LoRA/QLoRA fine-tuning pipeline (`docs/p2/SPEC.md`). P1 FastAPI kodu değişmez; P2 bağımsız bir `ml/` dizininde gelişir. P3'te P2 model çıktısı + P1 API birleştirilir.
