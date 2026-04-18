# P1 — Görev Listesi: Claude Code Session'ları

> Her görev = bir Claude Code session'ı.
> Bir session'ı bitirmeden bir sonrakine geçme.
> Her session sonunda `git commit` at.

---

## Nasıl kullanırsın?

1. VS Code'da `eduai-platform/` klasörünü aç
2. Claude Code'u aç (terminal ya da extension)
3. İlgili task'ın **"Claude Code'a ver"** bloğunu kopyala
4. Session bittikten sonra **"Bunu kendin yap"** adımlarını uygula
5. Bir sonraki task'a geç

---

## Task 0 — Repo yapısını kur ⏱ ~15 dk

**Bu task'ta öğreniyorsun:** Git flow, .gitignore, Python proje standardları.

### Claude Code'a ver:
```
Sen bir senior Python backend engineer'sın. Bana yardım edeceksin.

Görev: Aşağıdaki klasör yapısını oluştur. Her dosyada sadece gerekli boilerplate olsun (boş __init__.py, placeholder içerikli diğerleri).

Yapı:
eduai-platform/
├── services/
│   └── api/
│       ├── app/
│       │   ├── __init__.py
│       │   ├── main.py (FastAPI app instance, lifespan, sadece /health endpoint)
│       │   ├── routers/__init__.py
│       │   ├── schemas/__init__.py
│       │   ├── services/__init__.py
│       │   ├── core/
│       │   │   ├── __init__.py
│       │   │   └── config.py (pydantic-settings ile Settings class)
│       │   └── dependencies.py
│       ├── tests/
│       │   ├── __init__.py
│       │   └── conftest.py (TestClient fixture)
│       ├── Dockerfile (multi-stage build)
│       ├── requirements.txt
│       └── pyproject.toml (ruff config dahil)
├── docker-compose.yml
├── .github/workflows/ci.yml
├── .gitignore (Python + Docker + .env için)
└── README.md (placeholder)

Kurallar:
- Python 3.11, FastAPI 0.111+, Pydantic v2
- main.py'de lifespan pattern kullan (@asynccontextmanager)
- config.py'de pydantic_settings.BaseSettings kullan
- requirements.txt'e şunları ekle: fastapi, uvicorn[standard], pydantic-settings, structlog, python-multipart, httpx, pytest, pytest-asyncio, ruff
- .gitignore'a .env, __pycache__, .pytest_cache, *.pyc ekle
- docker-compose.yml: api servisi, port 8000, hot-reload aktif
```

### Bunu kendin yap (session bittikten sonra):
```bash
cd eduai-platform
git init
git add .
git commit -m "feat: initial project structure P1"
```

---

## Task 1 — Schemas katmanı ⏱ ~30 dk

**Bu task'ta öğreniyorsun:** Pydantic v2, field validation, enum, UUID.

### Claude Code'a ver:
```
Proje: eduai-platform/services/api/

Görev: schemas/ klasöründe 3 Pydantic v2 schema dosyası yaz.

--- schemas/questions.py ---
QuestionRequest:
  - question: str (min_length=5, max_length=500)
  - session_id: UUID
  - subject: SubjectEnum (aşağıda)
  - grade_level: int (ge=1, le=12)

SubjectEnum (str, Enum):
  matematik, fizik, kimya, biyoloji, tarih, edebiyat, ingilizce, genel

QuestionResponse:
  - question_id: UUID (default_factory=uuid4)
  - answer: str
  - confidence: float (ge=0.0, le=1.0, default=0.0)
  - sources: list[str] (default=[])
  - processing_time_ms: int
  - session_id: UUID

--- schemas/documents.py ---
DocumentUploadResponse:
  - document_id: UUID (default_factory=uuid4)
  - title: str (max_length=200)
  - subject: SubjectEnum (from questions.py'den import et)
  - grade_level: int
  - file_size_bytes: int
  - status: Literal["uploaded", "processing", "ready"]
  - created_at: datetime (default_factory=datetime.utcnow)

--- schemas/sessions.py ---
SessionResponse:
  - session_id: UUID
  - created_at: datetime
  - question_count: int (default=0)
  - last_activity: datetime
  - subjects_accessed: list[SubjectEnum] (default=[])

SessionCreateResponse:
  - session_id: UUID (default_factory=uuid4)
  - created_at: datetime (default_factory=datetime.utcnow)

Kurallar:
- Her dosyada model_config = ConfigDict(from_attributes=True) ekle
- Tüm field'lara açıklayıcı description= ekle (Swagger'da görünecek)
- from __future__ import annotations kullan
```

### Bunu kendin yap:
```bash
# Python'da test et:
cd services/api
python -c "from app.schemas.questions import QuestionRequest; print('OK')"
git add . && git commit -m "feat: add Pydantic schemas for questions, documents, sessions"
```

---

## Task 2 — Services katmanı ⏱ ~30 dk

**Bu task'ta öğreniyorsun:** Service pattern, in-memory storage, separation of concerns.

### Claude Code'a ver:
```
Proje: eduai-platform/services/api/

Görev: services/ klasöründe 3 service dosyası yaz. Bu katman iş mantığını içerir, HTTP bilmez.

--- services/question_service.py ---
class QuestionService:
  - __init__: in-memory store için dict[UUID, QuestionResponse] 
  - async process_question(request: QuestionRequest) -> QuestionResponse:
      1. UUID oluştur
      2. Mock cevap üret: f"[EduAI Mock] '{request.subject.value}' konusunda sorunuz alındı. P3 fazında gerçek AI cevabı buraya gelecek."
      3. processing_time_ms'i hesapla (time.monotonic ile)
      4. structlog ile log at: logger.info("question_processed", question_id=..., subject=...)
      5. QuestionResponse döndür
  - async get_question(question_id: UUID) -> QuestionResponse | None

--- services/document_service.py ---
class DocumentService:
  - in-memory store: dict[UUID, DocumentUploadResponse]
  - async save_document(title, subject, grade_level, file_size) -> DocumentUploadResponse:
      Mock: status="uploaded", metadata kaydet
  - async get_document(document_id) -> DocumentUploadResponse | None

--- services/session_service.py ---
class SessionService:
  - in-memory store: dict[UUID, dict] (session data)
  - async create_session() -> SessionCreateResponse
  - async get_session(session_id) -> SessionResponse | None
  - async record_question(session_id, subject) -> None:
      question_count += 1, last_activity güncelle, subject ekle

Kurallar:
- Her service sınıfı için structlog logger kullan
- Type hint'ler eksiksiz olsun
- Docstring yaz (her metoda 1 satır)
```

### Bunu kendin yap:
```bash
git add . && git commit -m "feat: add service layer with in-memory storage"
```

---

## Task 3 — Routers ⏱ ~45 dk

**Bu task'ta öğreniyorsun:** FastAPI router, Depends(), async endpoint, file upload.

### Claude Code'a ver:
```
Proje: eduai-platform/services/api/

Görev: routers/ klasöründe 3 router dosyası yaz. Router sadece HTTP katmanı — iş mantığı service'de.

--- dependencies.py ---
Singleton pattern ile service instance'ları:
  get_question_service() -> QuestionService
  get_document_service() -> DocumentService  
  get_session_service() -> SessionService
(Her request'te yeni instance değil, app genelinde tek instance)

--- routers/questions.py ---
router = APIRouter(prefix="/v1/questions", tags=["Questions"])

POST /ask:
  - QuestionRequest al
  - QuestionService'e delege et
  - QuestionResponse döndür
  - response_model=QuestionResponse
  - status_code=201

--- routers/documents.py ---
router = APIRouter(prefix="/v1/documents", tags=["Documents"])

POST /upload:
  - file: UploadFile, title: str, subject: SubjectEnum, grade_level: int Form field olarak
  - Dosya validasyonu: sadece .pdf ve .txt, max 10MB (10 * 1024 * 1024 bytes)
  - 10MB aşılırsa HTTPException(413, "File too large")
  - Yanlış format: HTTPException(415, "Only PDF and TXT files allowed")  
  - DocumentService'e delege et
  - DocumentUploadResponse döndür

--- routers/sessions.py ---
router = APIRouter(prefix="/v1/sessions", tags=["Sessions"])

POST /: yeni session oluştur → SessionCreateResponse (201)
GET /{session_id}: session getir → SessionResponse (404 if not found)

--- main.py güncelle ---
- Tüm router'ları include et
- /health endpoint'i ekle (mevcut varsa güncelle):
  {"status": "healthy", "version": settings.VERSION, "timestamp": datetime.utcnow()}
- CORS middleware ekle (allow_origins=["*"], development için)
- Swagger UI başlığı: "EduAI API", description ekle
```

### Bunu kendin yap:
```bash
# Sunucuyu başlat:
cd services/api
python -m uvicorn app.main:app --reload

# Tarayıcıda aç: http://localhost:8000/docs
# Tüm endpoint'ler görünüyor mu? Her birini manuel test et.

git add . && git commit -m "feat: add routers for questions, documents, sessions"
```

---

## Task 4 — Testler ⏱ ~45 dk

**Bu task'ta öğreniyorsun:** pytest-asyncio, TestClient, fixture pattern, test isolation.

### Claude Code'a ver:
```
Proje: eduai-platform/services/api/

Görev: tests/ klasöründe kapsamlı testler yaz.

--- tests/conftest.py ---
@pytest.fixture
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c

--- tests/test_questions.py ---
Test et:
1. test_ask_question_valid: geçerli payload → 201, question_id UUID, answer string
2. test_ask_question_too_short: 4 karakter soru → 422
3. test_ask_question_too_long: 501 karakter soru → 422
4. test_ask_question_invalid_subject: subject="coğrafya" → 422
5. test_ask_question_invalid_grade: grade_level=13 → 422
6. test_ask_question_invalid_grade_zero: grade_level=0 → 422

--- tests/test_documents.py ---
1. test_upload_valid_txt: geçerli .txt dosyası → 201, document_id var
2. test_upload_invalid_format: .exe dosyası → 415
3. test_upload_too_large: 11MB dosya → 413

--- tests/test_sessions.py ---
1. test_create_session: POST /v1/sessions/ → 201, session_id UUID
2. test_get_session_exists: oluştur, sonra GET → 200
3. test_get_session_not_found: rastgele UUID → 404

--- tests/test_health.py ---
1. test_health_ok: GET /health → 200, status=="healthy"

Kurallar:
- Her test fonksiyonu sadece 1 şeyi test etsin
- assert mesajları açıklayıcı olsun
- Dosya upload testlerinde io.BytesIO kullan
```

### Bunu kendin yap:
```bash
cd services/api
pytest tests/ -v

# Hepsi geçti mi? Geçmediyse hangisi neden?
git add . && git commit -m "test: add comprehensive test suite"
```

---

## Task 5 — Docker ⏱ ~30 dk

**Bu task'ta öğreniyorsun:** Multi-stage build, image boyutunu küçültme, container networking.

### Claude Code'a ver:
```
Proje: eduai-platform/services/api/

Görev: Production-ready Dockerfile yaz ve docker-compose.yml güncelle.

--- Dockerfile ---
Multi-stage build:
Stage 1 (builder): python:3.11-slim
  - WORKDIR /build
  - requirements.txt kopyala
  - pip install --user --no-cache-dir

Stage 2 (production): python:3.11-slim  
  - WORKDIR /app
  - builder'dan /root/.local kopyala
  - app/ kopyala
  - Non-root user oluştur: useradd -m appuser, USER appuser
  - HEALTHCHECK: curl -f http://localhost:8000/health || exit 1
  - PORT 8000 expose
  - CMD: uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2

--- docker-compose.yml ---
version: "3.9"
services:
  api:
    build: ./services/api
    ports: ["8000:8000"]
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

### Bunu kendin yap:
```bash
# Docker build test:
docker build -t eduai-api:local services/api/

# Image boyutunu gör:
docker images eduai-api:local

# Docker Compose ile çalıştır:
docker-compose up

# Başka terminal'de test et:
curl http://localhost:8000/health

git add . && git commit -m "feat: add multi-stage Dockerfile and docker-compose"
```

---

## Task 6 — GitHub Actions ⏱ ~20 dk

**Bu task'ta öğreniyorsun:** CI/CD, YAML workflow syntax, job dependencies.

### Claude Code'a ver:
```
Proje: eduai-platform/

Görev: .github/workflows/ci.yml dosyasını yaz.

Workflow adı: "CI — EduAI API"
Tetikleyiciler: push (tüm branch'lar), pull_request

Jobs:

Job 1 (lint):
  name: "Code Quality"
  runs-on: ubuntu-latest
  steps:
    - checkout
    - setup-python 3.11
    - pip install ruff
    - ruff check services/api/ --output-format=github
    - ruff format services/api/ --check

Job 2 (test):
  name: "Tests"
  runs-on: ubuntu-latest
  needs: lint
  steps:
    - checkout
    - setup-python 3.11
    - pip install -r services/api/requirements.txt
    - cd services/api && pytest tests/ -v --tb=short

Job 3 (docker-build):
  name: "Docker Build"
  runs-on: ubuntu-latest
  needs: test
  steps:
    - checkout
    - docker build -t eduai-api:ci services/api/
    - "Image size check: docker images eduai-api:ci --format 'Size: {{.Size}}'"
```

### Bunu kendin yap:
```bash
git add . && git commit -m "ci: add GitHub Actions workflow"
git push origin main

# GitHub'da repo'ya git → Actions tab'ı → workflow çalışıyor mu?
```

---

## Task 7 — README ⏱ ~20 dk

**Bu task'ta öğreniyorsun:** Teknik dokümantasyon yazma (CV için kritik).

### Claude Code'a ver:
```
Proje: eduai-platform/

Görev: README.md yaz. Dil: İngilizce. Hedef kitle: bir recruiter veya senior developer.

İçermesi gerekenler:

1. Başlık + rozet: CI/CD badge (GitHub Actions), Python versiyonu
2. Kısa açıklama (3 cümle): Ne bu proje? Neden var? Büyük resim ne?
3. Tech stack tablosu: Her teknoloji + neden kullanıldı
4. Quick Start:
   - Prerequisites: Docker, Python 3.11
   - git clone komutu
   - docker-compose up
   - http://localhost:8000/docs
5. Endpoint listesi: Tablo formatında (Method, Path, Description)
6. Örnek curl komutları: Her endpoint için
7. Proje yapısı: Açıklamalı klasör ağacı
8. Geliştirme notu: "Bu proje EduAI Platform'un P1'i. Sonraki fazlar: P2 (fine-tuning), P3 (agents), P4 (cloud deployment)"
9. Öğrenme hedefleri (Learning Goals): Bullet list

Ton: Professional, özlü. Markdown başlıkları H2 kullan. Code block'lar için uygun dil belirt.
```

### Bunu kendin yap:
```bash
git add . && git commit -m "docs: add comprehensive README"
git push origin main
```

---

## Proje tamamlandı mı? Kontrol listesi

```
[ ] docker-compose up → localhost:8000/docs açılıyor
[ ] Tüm 6 endpoint Swagger'da görünüyor
[ ] pytest → en az 10 test, hepsi yeşil
[ ] ruff check → sıfır hata
[ ] GitHub Actions → 3 job yeşil
[ ] Docker image < 200MB
[ ] README okunabilir, curl örnekleri çalışıyor
```

Hepsi tamamsa → **P2'ye geç.** P3 için P1 ayakta kalmalı.

---

## Sıkça karşılaşılan sorunlar

**"Import error: cannot import name X"**
→ `__init__.py` dosyalarını kontrol et. Her klasörde olmalı.

**"422 Unprocessable Entity" beklenmedik yerde**
→ Pydantic field validasyon hatası. Response'da `detail` alanına bak.

**"Docker build başarısız: requirements not found"**
→ `docker build` komutunu `services/api/` içinden değil, proje kökünden çalıştır.

**"pytest: no tests ran"**
→ Test dosyaları `test_` prefix'i ile başlamalı, fonksiyonlar da `test_` ile.

**GitHub Actions kırmızı**
→ Actions tab'ında hata mesajını oku. Genellikle lint hatası ya da import sorunu.
