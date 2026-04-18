# P1 Implementation Notes — SPEC sapmaları ve kararlar

> Bu dosya, P1 kodunun `docs/p1/SPEC.md`'den bilinçli olarak farklılaştığı her noktayı ve gerekçesini kaydeder.
>
> **Amaç:** Fresh Claude session'ları (özellikle P2'ye geçişte) kod-SPEC uyumsuzluğu algıladığında hallucinate etmek yerine **önce buraya bakmalı**.
>
> **Yaşam döngüsü:** Her P1 task'ı tamamlandığında bu dosya güncellenir. Task 7 (P1 final) sonunda SPEC.md bu notlar baz alınarak güncellenecek ve bu dosya arşive alınacak.

---

## Task 0 — Scaffolding

### README.md dokunulmadı
- **SPEC:** "README.md (placeholder)"
- **Uygulama:** Mevcut README.md korundu (proje özeti zaten placeholder niteliğinde vardı)
- **Sebep:** Üzerine yazmak bilgi kaybı olurdu
- **Etki:** Task 7'de production-quality README ayrıca yazılacak

---

## Task 1 — Schemas

### `datetime.utcnow` → `datetime.now(timezone.utc)`
- **SPEC:** `default_factory=datetime.utcnow` (documents.created_at, sessions.created_at)
- **Uygulama:** `Field(default_factory=lambda: datetime.now(timezone.utc))`
  (`schemas/documents.py:50`, `schemas/sessions.py:41, 54`)
- **Sebep:** `datetime.utcnow()` Python 3.12+ deprecated; naive datetime döner, timezone info yok
- **Etki:** JSON output ISO 8601 `+00:00` suffix'li olur — parsing/test assertion deterministik

### `default=[]` → `default_factory=list`
- **SPEC:** `sources: list[str] (default=[])`, `subjects_accessed: list[SubjectEnum] (default=[])`
- **Uygulama:** `Field(default_factory=list)`
  (`schemas/questions.py:71`, `schemas/sessions.py:47`)
- **Sebep:** Mutable default Python tuzağı — tüm instance'lar aynı listeyi paylaşır; Pydantic v2 zaten uyarır
- **Etki:** Davranış aynı (boş liste); state leak/race condition riski elimine

### `documents.grade_level` için `ge=1, le=12`
- **SPEC:** `grade_level: int` (bound yok)
- **Uygulama:** `Field(..., ge=1, le=12)` (`schemas/documents.py:39`)
- **Sebep:** `QuestionRequest.grade_level` ile tutarlılık — aynı semantik her yerde aynı kısıt taşımalı
- **Etki:** Invalid value'lar (0, 13) artık 422 → tutarlı validation

### `sources: list[str]` — tip belirlendi
- **SPEC:** Response örneğinde `"sources": []`; tip hint yoktu
- **Uygulama:** `list[str]` (`schemas/questions.py:71`)
- **Sebep:** Boş liste örneği URL/document_id listesini işaret ediyor
- **Etki:** Type safety; P3 RAG entegrasyonunda dokümantasyon açık

### `processing_time_ms` için `ge=0`
- **SPEC:** `processing_time_ms: int`
- **Uygulama:** `Field(..., ge=0)` (`schemas/questions.py:79`)
- **Sebep:** Negatif süre anlamsız, defensive coding
- **Etki:** Service'te hesaplama hatası varsa erken fail

---

## Task 2 — Services

### Internal `subjects_accessed: set`, dışa `sorted list`
- **SPEC:** `subjects_accessed: list[SubjectEnum]` (response schema)
- **Uygulama:** Store'da `set[SubjectEnum]`, `get_session()`'da `sorted(...)` ile list'e çevrilir
  (`services/session_service.py:36, 56`)
- **Sebep:** Set duplicate ekleme otomatik engeller (aynı subject iki kere record'lansa liste şişmez); `sorted()` deterministik output (test stability)
- **Etki:** API response hâlâ list; davranış SPEC response örneği (`["tarih", "matematik"]`) ile uyumlu

### `record_question` bilinmeyen session'da warning + return
- **SPEC:** Behavior belirtilmemiş
- **Uygulama:** `logger.warning(...)` sonra sessiz return (`services/session_service.py:82-88`)
- **Sebep:** Sessiz yutmak debug imkanı bırakmaz; exception fırlatmak router'ı kirletir
- **Etki:** 404 sorumluluğu router'da kalır; service visibility sağlar

### `datetime.utcnow` → timezone-aware (service içi)
- **Uygulama:** `datetime.now(timezone.utc)` (`services/session_service.py:38, 74`)
- **Sebep/Etki:** Aynı deprecation nedeni (Task 1)

---

## Task 3 — Routers + dependencies + main.py

### Singleton pattern: `@lru_cache`
- **SPEC:** "Singleton pattern ile service instance'ları" (implement yöntemi unspecified)
- **Uygulama:** `@lru_cache` decorator (`dependencies.py:27, 32, 37`)
- **Sebep:** Module-level global'den daha test-friendly (`.cache_clear()` ile reset); FastAPI resmi önerisi
- **Etki:** Aynı singleton davranışı; testlerde override imkanı

### Question router'da `session_service.record_question` çağrısı
- **SPEC:** "QuestionService'e delege et"
- **Uygulama:** Router hem `question_service.process_question` hem `session_service.record_question` çağırır (`routers/questions.py:33-36`)
- **Sebep:** `question_service`'in `session_service`'i bilmesi cross-service coupling yaratırdı; router = HTTP + orchestration layer
- **Etki:** Session aktivite kayıtları düzgün ilerler; service'ler birbirinden bağımsız kalır (P3'te DI-friendly)

### `UploadFile.size` None fallback
- **SPEC:** "max 10MB aşılırsa 413"
- **Uygulama:** `file.size is None` ise `await file.read()` ile ölç, sonra `seek(0)` (`routers/documents.py:53-63`)
- **Sebep:** Bazı client'lar `content-length` header göndermez → Starlette `UploadFile.size` None olur
- **Etki:** Edge case'lerde 413 doğru döner; büyük dosya bellek'e yüklenir (10MB cap ile sınırlı, CI için kabul)

### 201 Created — documents/sessions
- **SPEC:** Sadece `/v1/questions/ask` için "status_code=201" explicit
- **Uygulama:** `/v1/documents/upload` + `POST /v1/sessions/` da 201 (`routers/documents.py:29`, `routers/sessions.py:25`)
- **Sebep:** POST = resource creation → 201 REST-standard; tutarlılık test assertion'ını basitleştirir
- **Etki:** Test'lerde uniform 201 beklentisi; Swagger'da doğru semantic

### `/health` timestamp timezone-aware
- **SPEC:** `"timestamp": datetime.utcnow()`
- **Uygulama:** `datetime.now(timezone.utc).isoformat()` (`main.py:73`)
- **Sebep/Etki:** Aynı deprecation nedeni

### Swagger `description` genişletildi
- **SPEC:** "description ekle" (içerik belirtilmemiş)
- **Uygulama:** 3 satırlık açıklama — P1 kapsamı + P3 AI hedefi (`main.py:32-37`)
- **Sebep:** SPEC içerik vermemişti; README + docs/p1/CONCEPT'ten derlendi
- **Etki:** `/docs` recruiter/developer için daha anlamlı

---

## Task 4 — Tests

### `conftest.py` — `app` import fixture içinde
- **SPEC:** `with TestClient(app) as c: yield c` (import yeri unspecified)
- **Uygulama:** `from app.main import app` fixture fonksiyonu içinde, top-level değil (`tests/conftest.py:17`)
- **Sebep:** Collect-time'da main.py import edilmez → env/config sorunları fixture-use zamanına ötelenir
- **Etki:** Ortam problemlerinde hata mesajı daha anlamlı (hangi test çağırdığında patlıyor)

### `test_ask_question_invalid_subject` — `"coğrafya"` (ğ) korundu
- **SPEC:** `subject="coğrafya"` → 422
- **Uygulama:** Test SPEC'teki Türkçe ğ ile yazıldı (`tests/test_questions.py:68`)
- **Kullanıcı değişikliği:** `SubjectEnum`'a `COGRAFYA = "cografya"` (ASCII g) eklendi (`schemas/questions.py:20`)
- **Sebep:** `"coğrafya"` (ğ ile Unicode) ≠ `"cografya"` (ASCII g) — farklı string'ler, enum eşleşmiyor
- **Etki:** Test hâlâ 422 alır; kasıtlı bırakıldı — Türkçe karakter edge case'i olarak değerli

---

## Task 5 — Docker

### `/root/.local` → `/home/appuser/.local`
- **SPEC:** `COPY --from=builder /root/.local /root/.local`
- **Uygulama:** `COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local` (`services/api/Dockerfile:33`)
- **Sebep:** SPEC Stage 2'de `USER appuser`'a geçiyor, ama `/root/` dizini 0700 (root'a özel) → appuser import edemez, `uvicorn` çalışmaz. Non-root user SPEC'in kendi gerektirdiği bir şey; kırılmaması için hedef path appuser home'una taşındı. `PATH` da `/home/appuser/.local/bin` olacak şekilde güncellendi.
- **Etki:** Container non-root çalışır + Python paketleri düzgün import edilir. Davranış SPEC niyetiyle birebir uyumlu.

### HEALTHCHECK için `curl` kurulumu
- **SPEC:** `HEALTHCHECK: curl -f http://localhost:8000/health`
- **Uygulama:** `apt-get install -y --no-install-recommends curl` + apt cache temizliği (`services/api/Dockerfile:22-25`)
- **Sebep:** `python:3.11-slim` imajında `curl` default yok — SPEC'teki healthcheck doğrudan çalışmaz
- **Etki:** Image ~10MB büyür; alternatif `python -c "urllib.request..."` olurdu ama SPEC + docker-compose.yml her ikisi de `curl` istiyor → tutarlılık için kurulum tercih edildi

### HEALTHCHECK parametrik timing
- **SPEC:** Sadece komut verilmiş, timing parametreleri yok
- **Uygulama:** `--interval=30s --timeout=10s --start-period=5s --retries=3` (`services/api/Dockerfile:43-44`)
- **Sebep:** `--start-period` uvicorn init sırasında "unhealthy" false-positive'leri önler; kalan parametreler docker-compose.yml ile aynı → Dockerfile standalone kullanılsa da tutarlı davranır
- **Etki:** Container startup'ta 5 saniyelik grace period; sonrasında 30s interval

### `useradd -m -u 1000 appuser` — UID explicit
- **SPEC:** `useradd -m appuser` (UID belirtilmemiş)
- **Uygulama:** `useradd -m -u 1000 appuser` (`services/api/Dockerfile:29`)
- **Sebep:** UID 1000 host'taki ilk kullanıcının default UID'i → volume mount'ta file ownership sorunu çıkmaz (dev'de bind mount uyumluluğu)
- **Etki:** Host user'ın dokunduğu dosyalar container'da appuser ownership'de görünür

### `.dockerignore` eklendi (yeni dosya)
- **SPEC:** İstenmedi
- **Uygulama:** `services/api/.dockerignore` — `tests/`, `.git`, `__pycache__`, `.env`, IDE config'leri dışarıda tutulur
- **Sebep:** Production imajında test/dev artifact'leri olmamalı; build context küçülür, build süresi kısalır
- **Etki:** Image boyutu azalır; test'ler lokalde/CI'de çalışır (spec zaten öyle yapıyor), imajda gereksiz

### `syntax=docker/dockerfile:1.6` directive
- **SPEC:** İstenmedi
- **Uygulama:** Dockerfile ilk satırı (`services/api/Dockerfile:1`)
- **Sebep:** BuildKit modern syntax özelliklerini etkinleştirir (`--chown` COPY, heredoc vs.)
- **Etki:** Yerel Docker Desktop'ta default; eski docker daemon'larda uyarı çıkarabilir

### Dev/prod requirements ayrımı (`requirements-dev.txt` eklendi)
- **SPEC:** Tek `requirements.txt` — fastapi, uvicorn, pydantic-settings, structlog, python-multipart, **httpx, pytest, pytest-asyncio, ruff** bir arada
- **Uygulama:** `requirements.txt` sadece prod deps (5 paket) + `requirements-dev.txt` (`-r requirements.txt` + httpx/pytest/pytest-asyncio/ruff)
- **Sebep:** İlk build sonrası image boyutu 340MB geldi → SPEC kontrol listesindeki `<200MB` hedefi aşıldı. Dev deps (pytest ~10MB, ruff ~25MB Rust binary, httpx ~8MB) production imajında hiçbir iş görmez. Ayrıca: az paket = küçük CVE surface, daha hızlı K8s pod startup.
- **Etki:** Image 340MB → **284MB** (gerçek ölçüm). **CI workflow güncellendi** → `pip install -r requirements-dev.txt` (prod deps otomatik inherit). Dockerfile değişmedi (zaten `requirements.txt` kullanıyordu). Container içinde `pytest`/`ruff` komutu bulunmaz (intended — testler CI/lokalde çalışır).

### Image boyutu <200MB hedefi karşılanmadı (P1'de kabul)
- **SPEC (kontrol listesi):** `[ ] Docker image < 200MB`
- **Uygulama:** 284MB, daha fazla trim yapılmadı
- **Sebep:** Kalan boyut python:3.11-slim base (~130MB) + pydantic v2 Rust core (~40MB) + uvicorn[standard] transitive (~50MB) + curl + starlette/anyio. <200MB için gerçekçi tek yol `python:3.11-alpine` (musl libc) — ama pydantic v2 alpine wheel yayınlamıyor (source build gerekir), P2'de PyTorch alpine'da problematik. Alternatif hafif kazanımlar (`uvicorn` extras trim + `pip --no-compile`) tahmini ~40MB düşüş → 240MB; hâlâ hedef altı değil, sapma/risk oranı düşük verim.
- **Etki:** P1 kabul kriteri karşılanmadı ama bilinçli bir karar. P4 (K8s/AWS) geçişinde yeniden değerlendirilecek: distroless image, multi-stage scratch, veya dependency audit ile 200MB altına gidilebilir.

---

## Task 6 — GitHub Actions CI

### Test job `requirements.txt` → `requirements-dev.txt`
- **SPEC:** `pip install -r services/api/requirements.txt`
- **Uygulama:** `pip install -r services/api/requirements-dev.txt` (`.github/workflows/ci.yml:44`)
- **Sebep:** Task 5'te dev/prod requirements ayrımı yapıldı — `pytest` artık `requirements.txt`'te yok. SPEC'teki komutla test job `ModuleNotFoundError: No module named 'pytest'` hatası verirdi. `requirements-dev.txt` `-r requirements.txt` ile prod deps'i inherit eder (tek komut, tam env).
- **Etki:** Test job yeşil geçer. SPEC güncellenince (Task 7) bu komut `requirements-dev.txt` olarak kalmalı.

### `setup-python` cache: "pip" eklendi
- **SPEC:** `cache` parametresi istenmedi
- **Uygulama:** `cache: "pip"` lint + test job'larında (`.github/workflows/ci.yml:20, 43`)
- **Sebep:** GitHub Actions `actions/setup-python@v5` pip wheel cache desteği veriyor; ikinci run'dan itibaren install süresi dakikalardan saniyelere iner. Zararsız, opt-in feature.
- **Etki:** CI run süresi ~3-4dk → ~1-2dk (cache hit'te). PR iteration hızlanır.

### Docker image size check: `--format '{{.Size}}'` kaldırıldı
- **SPEC:** `docker images eduai-api:ci --format 'Size: {{.Size}}'`
- **Uygulama:** `docker images eduai-api:ci` (template silindi) (`.github/workflows/ci.yml:78`)
- **Sebep:** GitHub Actions workflow parser `{{ ... }}` ifadelerini expression syntax olarak yorumluyor → single-quoted olsa bile `Invalid workflow file` hatası (YAML syntax error L78). Bu, Actions'ın template preprocessor'ının bilinen davranışı.
- **Etki:** Çıktı formatı farklı — tablo halinde tüm sütunlar (REPOSITORY/TAG/ID/CREATED/SIZE) görünür, SIZE bilgisi yine mevcut. SPEC niyeti (build log'da image boyutu görünür) korundu.
- **Alternatif:** `${{ '{{.Size}}' }}` escape pattern denenebilirdi ama kırılgan; plain command daha okunaklı.

### Lint job'da `pip install ruff` — version pinlenmedi
- **SPEC:** `pip install ruff` (version yok)
- **Uygulama:** SPEC'e sadık, unpinned (`.github/workflows/ci.yml:22`)
- **Sebep:** SPEC basitlik istedi; ruff her upgrade'de yeni rule aktif edebilir → CI kırılabilir
- **Etki:** Küçük CI kırılma riski; yaşandığında `pip install ruff==X.Y.Z` ile pinlenecek. P4'te (production) muhtemelen fix'lenir.

---

## Task 6.5 — Ruff compliance düzeltmeleri

İlk CI koşumundan önce lokalde `ruff check` 24 uyarı verdi. Modern Python / FastAPI best practice'e uyum için uygulanan düzeltmeler (hiçbirinin davranışsal etkisi yok; testler yeşil geçmeye devam etmeli):

### UP017 — `datetime.UTC` alias
- **Kapsam:** 5 yer (`app/main.py`, `app/schemas/documents.py`, `app/schemas/sessions.py`, `app/services/session_service.py` x2)
- **Değişim:** `from datetime import timezone` + `timezone.utc` → `from datetime import UTC` + `UTC`
- **Sebep:** Python 3.11+ `datetime.UTC` kısa alias sağlıyor; daha okunabilir, ruff default rule
- **Etki:** Davranış birebir aynı; import satırı kısaldı

### UP042 — `SubjectEnum(StrEnum)`
- **Kapsam:** `app/schemas/questions.py` — `class SubjectEnum(str, Enum)` → `class SubjectEnum(StrEnum)`
- **Sebep:** Python 3.11+ `StrEnum` tam olarak `(str, Enum)` çoklu miras pattern'ini tekleştirir; daha net niyet
- **Uyumluluk:** `.value`, eşitlik, JSON serialization aynı. `str(enum)` davranışı farklı (StrEnum raw value döner) ama kodumuzda kullanılmıyor — kontrol edildi.

### B008 — FastAPI `Annotated` pattern
- **Kapsam:** 6 parametre (`routers/questions.py`, `routers/documents.py`, `routers/sessions.py`)
- **Değişim:**
  - Önce: `service: T = Depends(fn)` — `Depends()` çağrısı argument default'ta
  - Sonra: `service: Annotated[T, Depends(fn)]` — çağrı type annotation içinde
- **Sebep:** B008 generic bir kural (function call as default tehlikeli — sadece bir kez eval edilir). FastAPI 0.95+ **resmi olarak önerilen modern syntax** zaten bu. Ruff + best practice aynı yöne bakıyor.
- **Etki:** Davranış + OpenAPI + Swagger UI birebir aynı. Router imzaları biraz daha uzun ama niyet daha net.

### E501 — Uzun test assertion mesajları
- **Kapsam:** 11 satır (`tests/test_questions.py`, `tests/test_documents.py`, `tests/test_sessions.py`)
- **Değişim:** Assertion mesajları multi-line parantez ile bölündü:
  ```python
  assert response.status_code == 422, (
      f"4 karakterlik soru 422 dönmeli, gelen {response.status_code}"
  )
  ```
- **Sebep:** `pyproject.toml`'daki `line-length = 100` kuralı
- **Etki:** Mesaj içeriği korundu (fail debug değeri aynı); satır sayısı marjinal arttı

**Not:** UP017 ve UP042 modern Python'a doğal geçiş — geri dönülmemeli. Annotated pattern FastAPI'nin yeni standardı; P2/P3'te yeni router yazarken bu pattern kullanılmalı (eski `= Depends()` yazılmamalı).

---

## Özet — Tekrar eden kararlar

İki pattern sık görülüyor, P2/P3'te otomatik uygulanmalı:

1. **`datetime.utcnow` → `datetime.now(timezone.utc)`** — her yerde
2. **`default=[]` / `default={}` → `default_factory=list/dict`** — her Pydantic Field'da

---

## P1 Final (Task 7 sonrası) — SPEC.md güncelleme planı

1. Yukarıdaki sapmaları SPEC.md'nin ilgili bölümlerine inline işle
2. SPEC.md'nin eski hali git history'de korunur
3. Bu dosya `IMPLEMENTATION_NOTES_ARCHIVED.md` olarak yeniden adlandırılır veya SPEC header'ına referans düşülür
4. P2 Claude'una sadece güncellenmiş SPEC.md + kod verilir (iki dosya arası tutarlılık yükü yok)
