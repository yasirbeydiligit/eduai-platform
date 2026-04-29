# P3 — Implementation Notes (Sapmalar)

> Spec'ten bilinçli ayrılan her karar burada gerekçesiyle kayıt altında.
> P1 + P2 pattern'inin devamı (P2'de 31 sapma). Task'lar ilerledikçe ek alır.
> Faz sonu inline callout'larla SPEC.md'ye taşınır + bu dosya
> `IMPLEMENTATION_NOTES_ARCHIVED.md` olur.
>
> **As-of:** 2026-04-29 (P3 Task 0 başlangıcı)

---

## Task 0 — Yapı + Qdrant smoke test

### Sapma 1 — Qdrant `latest` tag yerine sabit version pin

**Spec:** `image: qdrant/qdrant:latest`
**Uygulanan:** `image: qdrant/qdrant:v1.12.4`
**Gerekçe:** `latest` reproducibility'ı kırar; bir gün çalışan compose ertesi
gün bozulabilir (Qdrant minor version'ında collection schema/serialization
değişiklikleri olmuştur). Floor pin yaklaşımı (P2 ml/requirements.txt
strategy) docker imajları için sabit pin'e dönüşüyor — sürüm yükseltme
bilinçli karar olur. v1.12.4 P3 spec yazıldığı tarihte stable; gerekirse
test edilip yükseltilir.

### Sapma 2 — `api → qdrant` `depends_on` Task 0'da eklenmedi

**Spec:** Implicit; SPEC.md docker-compose örneğinde belirtilmemiş.
**Uygulanan:** API servisi qdrant'a bağımlı değil (Task 0'da). `QDRANT_URL`
ENV var olarak hazır.
**Gerekçe:** P1 API henüz qdrant'ı kullanmıyor; `depends_on` zorlanırsa
standalone API geliştirme için her seferinde qdrant container'ı başlatmak
zorunda kalınır. Task 5'te `/ask/v2` endpoint'i eklenince `depends_on:
qdrant` + condition: `service_healthy` eklenecek.

### Sapma 3 — agents/ paketinde `__init__.py` ile importable iskelet

**Spec:** Klasör yapısı listesi feature dosyalarını (`indexer.py`, `nodes.py`
vb.) sayar.
**Uygulanan:** Her klasörde `__init__.py` (kısa docstring), feature .py'leri
ilgili task'larda yazılacak (Task 1-6).
**Gerekçe:** Boş `indexer.py`, `nodes.py` vs. yaratmak commit gürültüsü ve
pytest collection ile yanlış import hatalarına yol açar. `__init__.py` paket
yapısını kurmaya yeter; ayrıca boş .py'ler "yazılmış ama implement edilmemiş"
sinyali yanıltıcı. Task ilerledikçe gerçek dosyalar eklenir.

### Sapma 4 — test_connection.py `vector_size=384` dummy

**Spec:** Boyut belirtilmemiş; sadece "test verisi yükle".
**Uygulanan:** `VECTOR_SIZE = 384` constant.
**Gerekçe:** Embedding modeli **Task 1'de seçilecek** (CONCEPT.md `bge-m3`
1024 / `multilingual-e5-large` 1024 / `emrecan` 768 benchmark öneriyor).
384 sentence-transformers MiniLM ailesi yaygın boyutu, smoke test için
nötr seçim. Gerçek collection Task 1'de seçilen modelin çıktı boyutuyla
yeniden oluşturulacak.

### Sapma 5 — Qdrant healthcheck eklendi (P1 pattern'i)

**Spec:** Healthcheck önerisi yok.
**Uygulanan:** `wget --spider http://localhost:6333/readyz` healthcheck +
`start_period: 10s`.
**Gerekçe:** P1 api servisi healthcheck pattern'ini izliyor. Task 5'te
`api → depends_on: qdrant: condition: service_healthy` eklendiğinde
zorunlu. Qdrant 1.9+ `/readyz` endpoint'i collection rebuild'i tamamlandı
sinyali; basit `/livez` collection ready demeyebilir.

### Sapma 6 — `pytest-asyncio`, `python-dotenv`, `pydantic-settings`, `structlog`
**ek bağımlılıklar (SPEC.md requirements.txt'inde yoktu)**

**Spec:** `pytest>=8.3` + `ruff>=0.6` dev için listelenmiş.
**Uygulanan:** Yukarıdaki 4 paket eklendi.
**Gerekçe:**
- `pytest-asyncio`: graph/nodes.py tüm async; Task 6 testleri için zorunlu.
- `python-dotenv`: ANTHROPIC_API_KEY, QDRANT_URL .env yüklemesi (P2 pattern'i).
- `pydantic-settings`: P1 `services/api/app/core/config.py` ile uyum (Task 5 entegrasyonunda settings sınıfı paylaşımı kolay).
- `structlog`: P1 logging stack'i; agent node'larında log emit etmek için.

---

## Task 1+ için flag'lenmiş açık konular

(Henüz uygulanmadı; ilgili task başında karar verilecek.)

### F-1 (Task 1) — Embedding modeli seçimi: SPEC vs CONCEPT çelişkisi

**SPEC.md:** `emrecan/bert-base-turkish-cased-mean-nli-stsb-tr` (2021)
**CONCEPT.md § 2:** "2-3 modeli kıyaslamaya değer — `BAAI/bge-m3`,
`intfloat/multilingual-e5-large`, `Alibaba-NLP/gte-multilingual-base`"

Plan: Task 1'de mini benchmark (50 soru × 3 model × cosine similarity ortalama)
~30 dk. Sonuç IMPLEMENTATION_NOTES'a Sapma olarak kaydedilir.

### F-2 (Task 4) — `crewai>=0.80.0` floor pin tarihi

SPEC 2024 yazılmış; CrewAI Apr 2026 itibariyle 0.150+ olabilir, breaking
changes muhtemel (Agent imzası, Task tanımı, LLM provider config).
Plan: Task 4 başında güncel sürüme bakılır + breaking change diff incelenir.

### F-3 (Task 1/3) — `langchain-anthropic`, `langchain-qdrant` integrations

SPEC requirements.txt'te yok ama Task 3'te Anthropic chat model wrapper +
Task 1'de Qdrant ↔ langchain RetrievalQA bağı için gerekebilir.
Plan: Task 1'de qdrant-client direkt kullanılırsa skip; langchain
RetrievalQA'ya geçilirse `langchain-qdrant>=0.2` eklenir. Anthropic Task 3'te
`langchain-anthropic>=0.3` eklenecek.

### F-4 (Task 0 → 3 → 5) — P2 adapter taşıma yolu

P3_HANDOFF.md § 2 üç seçenek: Drive (Colab dev), HF Hub, lokal indir.
Task 3 LLM_BACKEND=qwen3-local'a geçtiğinde karar verilecek. Geliştirme
Anthropic API ile ilerlediği için Task 0-4 boyunca açık kalabilir.

---

## Disiplin pattern'leri (P1 + P2'den devralındı)

Hatırlatma — bu defterde her sapma için:
1. **Spec ne diyordu?**
2. **Ne uygulandı?**
3. **Neden?** (gerekçe + alternatif düşünüldü mü)

P2 finalize sonrası bu dosya `IMPLEMENTATION_NOTES_ARCHIVED.md` olur,
SPEC.md'ye inline callout'lar eklenir.
