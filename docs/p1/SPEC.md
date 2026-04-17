# P1 — EduAI API: Proje Spesifikasyonu

> Bu dosyayı Claude Code'a veriyorsun. "Bunu yap" değil, "bu spec'e göre production-ready kod yaz" diyorsun.

---

## Proje özeti

**Ne:** Türkçe eğitim içerikleri için AI-powered soru-cevap platformunun backend API'si.  
**Amaç:** Sonraki fazlarda (fine-tuning, agent sistemi, cloud deploy) kullanılacak sağlam temel.  
**Şu anki kapsam:** FastAPI + Docker + GitHub Actions. AI yok henüz — sadece mimari.

---

## Teknik gereksinimler

### Stack
- Python 3.11+
- FastAPI 0.111+
- Pydantic v2
- Uvicorn (ASGI server)
- pytest + httpx (testler için)
- ruff (linter + formatter)
- Docker + Docker Compose
- GitHub Actions

### Klasör yapısı — tam olarak bu şekilde oluştur

```
eduai-platform/
├── services/
│   └── api/
│       ├── app/
│       │   ├── __init__.py
│       │   ├── main.py
│       │   ├── routers/
│       │   │   ├── __init__.py
│       │   │   ├── questions.py
│       │   │   ├── documents.py
│       │   │   └── sessions.py
│       │   ├── schemas/
│       │   │   ├── __init__.py
│       │   │   ├── questions.py
│       │   │   ├── documents.py
│       │   │   └── sessions.py
│       │   ├── services/
│       │   │   ├── __init__.py
│       │   │   ├── question_service.py
│       │   │   ├── document_service.py
│       │   │   └── session_service.py
│       │   ├── core/
│       │   │   ├── __init__.py
│       │   │   ├── config.py
│       │   │   └── logging.py
│       │   └── dependencies.py
│       ├── tests/
│       │   ├── __init__.py
│       │   ├── conftest.py
│       │   └── test_questions.py
│       ├── Dockerfile
│       ├── requirements.txt
│       └── pyproject.toml
├── docker-compose.yml
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore
└── README.md
```

---

## Endpoint spesifikasyonları

### POST /v1/questions/ask

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
  "answer": "Osmanlı İmparatorluğu 1299 yılında Osman Bey tarafından kurulmuştur. [Mock cevap - AI entegrasyonu P3'te gelecek]",
  "confidence": 0.0,
  "sources": [],
  "processing_time_ms": 42,
  "session_id": "uuid"
}
```

**Validasyon kuralları:**
- `question`: min 5, max 500 karakter
- `session_id`: geçerli UUID4
- `subject`: enum → `["matematik", "fizik", "kimya", "biyoloji", "tarih", "edebiyat", "ingilizce", "genel"]`
- `grade_level`: int, 1-12 arası

---

### POST /v1/documents/upload

**Request:** `multipart/form-data`
- `file`: PDF veya TXT, max 10MB
- `subject`: string (yukarıdaki enum)
- `title`: string, max 200 karakter
- `grade_level`: int

**Response:**
```json
{
  "document_id": "uuid",
  "title": "...",
  "subject": "matematik",
  "grade_level": 9,
  "file_size_bytes": 102400,
  "status": "uploaded",
  "created_at": "2026-04-13T10:00:00Z"
}
```

**Not:** Şu an dosyayı sadece metadata olarak kaydet (in-memory). P3'te Qdrant'a vektörize edilecek.

---

### GET /v1/sessions/{session_id}

**Response:**
```json
{
  "session_id": "uuid",
  "created_at": "2026-04-13T10:00:00Z",
  "question_count": 5,
  "last_activity": "2026-04-13T10:30:00Z",
  "subjects_accessed": ["tarih", "matematik"]
}
```

**POST /v1/sessions/** — yeni session oluştur, UUID döndür.

---

### GET /health

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": "2026-04-13T10:00:00Z",
  "services": {
    "api": "healthy",
    "storage": "healthy"
  }
}
```

---

## Mimari kurallar — bunlara kesinlikle uy

### 1. Router sadece HTTP katmanı

```python
# YANLIŞ:
@router.post("/ask")
async def ask_question(request: QuestionRequest):
    # iş mantığı direkt burada
    answer = f"Mock: {request.question}"
    return QuestionResponse(answer=answer)

# DOĞRU:
@router.post("/ask")
async def ask_question(
    request: QuestionRequest,
    service: QuestionService = Depends(get_question_service)
):
    return await service.process_question(request)
```

### 2. Settings — environment variable'lardan oku

```python
# core/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "EduAI API"
    VERSION: str = "0.1.0"
    DEBUG: bool = False
    MAX_UPLOAD_SIZE_MB: int = 10
    
    class Config:
        env_file = ".env"
```

### 3. Logging — print() yok, logger var

```python
import structlog
logger = structlog.get_logger()
logger.info("question_processed", question_id=str(question_id), duration_ms=42)
```

### 4. Error handling — HTTPException değil, custom exception handler

```python
# Tüm beklenmedik hatalar için global handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error("unhandled_exception", error=str(exc))
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
```

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

---

## Dockerfile — multi-stage build

```dockerfile
# Stage 1: builder
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: production
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY app/ app/
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

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
    volumes:
      - ./services/api/app:/app/app  # hot reload için
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## GitHub Actions — ci.yml

Her push'ta çalışacak:
1. Ruff ile lint kontrolü
2. pytest ile test koşumu
3. Docker image build (push değil, sadece build başarılı mı?)

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r services/api/requirements.txt
      - run: ruff check services/api/
      - run: pytest services/api/tests/ -v
  
  docker:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - run: docker build -t eduai-api:ci services/api/
```

---

## Test gereksinimleri

En az şu testler yazılmalı:

```python
# test_questions.py
def test_ask_question_valid():
    """Geçerli soru → 200 + question_id döner"""

def test_ask_question_too_short():
    """5 karakterden kısa soru → 422 Validation Error"""

def test_ask_question_invalid_subject():
    """Enum dışı subject → 422"""

def test_health_endpoint():
    """Health check → 200 + status: healthy"""
```

---

## README.md gereksinimleri

README şunları içermeli:
- Projenin ne olduğu (2-3 cümle, İngilizce)
- Hızlı başlangıç: `docker-compose up`
- Endpoint listesi ve örnek curl komutları
- Proje yapısı açıklaması
- "Bu projenin amacı nedir?" — öğrenme hedeflerini yaz

---

## Teslim kriterleri

Bu proje tamamlandığında:
- [ ] `docker-compose up` komutu çalışıyor, `localhost:8000/docs` açılıyor
- [ ] Tüm endpoint'ler `/docs`'ta görünüyor
- [ ] `pytest` geçiyor (en az 4 test)
- [ ] `ruff check` sıfır hata
- [ ] GitHub Actions yeşil
- [ ] README okunabilir ve İngilizce
