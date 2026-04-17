# P1 — Servis İskeleti: Konsept Kılavuzu

> Bu dosyayı Claude Code'a vermeden önce oku. Kör kopyalamak değil, **ne yaptığını anlayarak** yazmak istiyoruz.

---

## Bu proje neden var?

Sonraki 3 projede (fine-tuning, agent sistemi, cloud deploy) hep aynı soruyla karşılaşacaksın:
**"Bunu dışarıya nasıl açarım?"**

Cevap her seferinde aynı: bir HTTP API. P1'de bu API'yi bir kez, doğru şekilde kuruyorsun. Sonraki projelerde sadece yeni endpoint ekleyeceksin.

---

## Temel kavramlar — bunları anlamadan SPEC'e geçme

### 1. FastAPI neden?

Flask ya da Django değil FastAPI çünkü:
- **Otomatik dokümantasyon** → `/docs` endpoint'i. Recruiter'a "işte API'm" diyebilirsin.
- **Pydantic entegrasyonu** → input/output şemaları otomatik valide edilir. Production'da tip hatası olmaz.
- **Async-first** → AI modelleri yavaş. Async olmayan bir API, model cevap verene kadar tüm diğer istekleri bloklar.
- **Type hints** → Python'un en güçlü özelliği. Modern kod böyle yazılır.

**Şunu anla:** FastAPI bir web framework'ü değil, bir **API framework'ü**. Frontend yok, sadece JSON konuşuyor.

---

### 2. Docker neden?

"Bende çalışıyor" problemi. Sen Mac'te geliştirdin, recruiter Linux'ta çalıştırdı, şirket Windows server'da deploy etti — hepsi farklı davranır.

Docker şunu söyler: **"Uygulamanı kutunun içine koy. Kutu her yerde aynı çalışır."**

Bu kutu = **image**. Çalışan kutu = **container**.

```
Dockerfile → image tarifi
docker build → image üretir
docker run → container başlatır
docker-compose → birden fazla container'ı birlikte yönetir
```

**Şunu anla:** Production'da hiçbir şey "bare metal"da çalışmaz. Her şey container'da. Docker bilmeden senior ilanına başvuramazsın.

---

### 3. GitHub Actions neden?

Her `git push`'ta otomatik olarak:
1. Testleri koştur
2. Kodu lint'le (PEP8 kontrolü)
3. Docker image build et

Buna **CI (Continuous Integration)** denir. Şirketlerde kod review'dan önce CI geçmek zorunda. CV'de "CI/CD kurdum" yazabilmek için bunu bilmen lazım.

---

### 4. Pydantic neden önemli?

```python
# Bu çöp:
def ask_question(data: dict):
    question = data["question"]  # KeyError riski var
    
# Bu production-ready:
class QuestionRequest(BaseModel):
    question: str
    session_id: UUID
    language: Literal["tr", "en"] = "tr"

def ask_question(data: QuestionRequest):
    # data.question garantili var, tip garantili str
```

FastAPI + Pydantic = otomatik validasyon + otomatik dokümantasyon. İkisi birlikte olmazsa olmaz.

---

### 5. Proje yapısı neden böyle?

```
app/
├── main.py          ← sadece FastAPI instance ve lifespan
├── routers/         ← endpoint grupları (her domain ayrı dosya)
├── schemas/         ← Pydantic modelleri
├── core/            ← config, settings, constants
├── services/        ← iş mantığı (endpoint'ten ayrı!)
└── dependencies.py  ← dependency injection
```

**En önemli prensip:** `routers/` sadece HTTP katmanı. Asıl iş `services/`'de. Bu ayrımı yapmazsan kod test edilemez hale gelir.

---

### 6. Dependency Injection (DI) nedir?

```python
# DI olmadan (kötü):
@app.get("/questions")
async def get_questions():
    db = Database(url="...")  # her request'te yeni bağlantı
    
# DI ile (iyi):
async def get_db():
    db = Database(url=settings.DATABASE_URL)
    yield db
    await db.close()

@app.get("/questions")
async def get_questions(db: Database = Depends(get_db)):
    # db dışarıdan inject edildi, test ederken mock'lanabilir
```

FastAPI'de `Depends()` = DI. Test yazabilmek için şart.

---

## EduAI API — ne servis ediyor?

```
POST  /v1/questions/ask        → Soru gönder, (şimdilik) mock cevap al
POST  /v1/documents/upload     → PDF/TXT yükle, metadata kaydet
GET   /v1/sessions/{id}        → Öğrenci oturumu sorgula
GET   /health                  → Sistem sağlık kontrolü
GET   /docs                    → Swagger UI (FastAPI otomatik üretir)
```

Şu an hiçbir endpoint gerçek AI kullanmıyor. Ama mimarisi hazır. P3'te `/v1/questions/ask`'a AI agent bağlayacaksın. Şimdi sadece foundation kuruluyor.

---

## Başlamadan önce sorular

Bu konsepti okuduktan sonra şunları cevaplayabilmelisin:

1. `async def` neden önemli? `def` kullansam ne olur?
2. Pydantic modeli ile Python dataclass farkı ne?
3. Docker image ile container farkı ne?
4. CI neden kod yazmaktan ayrı bir kavram?
5. `services/` katmanını neden `routers/`'dan ayırıyoruz?

Cevapları bilmiyorsan bu dosyanın ilgili bölümünü tekrar oku. Sonra SPEC.md'ye geç.
