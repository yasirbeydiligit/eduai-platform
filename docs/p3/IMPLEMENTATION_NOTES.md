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

## Task 1 — RAG bileşenleri (devam ediyor)

### Sapma 7 — Embedding modeli: `intfloat/multilingual-e5-large` (SPEC: emrecan)

**Spec:** `emrecan/bert-base-turkish-cased-mean-nli-stsb-tr` (2021, 768-dim)
**Uygulanan:** `intfloat/multilingual-e5-large` (1024-dim, instruction prefix support)

**Süreç:** İki aşamalı empirical benchmark (`agents/scripts/embedding_benchmark*.py`).

**1. tur** — 9 paragraf × 5 soru, 4 model (BAAI/bge-m3, multilingual-e5-large,
Alibaba-NLP/gte-multilingual-base, emrecan):
- 3 model %100 tie (test ayrım yapamadı; anahtar kelime ortaklığı yüksekti)
- bge-m3 encode latency 65 sn (multi-vector hesabı ağır) → elendi
- gte-multilingual-base hata: ST 5.4.1 + custom modeling.py ABI çakışması
  (`index out of bounds 0..93`) → elendi
- e5-large vs emrecan tie kaldı

**2. tur (HARD)** — 21 paragraf × 10 paraphrase-ağır soru, distractor'larla
(aynı konuda yakın yanlışlar: 5 Osmanlı reformu, Newton+Kepler+Galilei,
Servet-i Fünun ↔ Tanzimat edebiyatı):
- Her iki model %100 top-1 (10/10), %100 top-3
- Skor margin'i belirleyici fark:
  - **e5-large** avg score 0.859, min 0.832 (geniş margin)
  - **emrecan** avg score 0.680, min 0.580 (Q3/Q4 distractor-yakın çiftlerde
    skor 0.58'e iniyor → büyük korpusta cross-distractor kayıp riski)
- Latency: emrecan 7x daha hızlı (0.014 vs 0.105 sn/metin) ama indexing
  offline + query-time ~6 ms fark kullanıcı algılaması altında.

**Gerekçe:**
1. Confidence margin: LangGraph validator node'u "score < threshold → retry"
   pattern'i ile çalışacak (Task 3); e5-large'da meaningful threshold (~0.7)
   mümkün, emrecan'da tüm scoreler 0.5-0.8 arasında dar dağılmış → threshold
   anlamsız.
2. Generalization: emrecan 2021, küçük/Türkçe-only korpusta iyi ama
   production'da binlerce chunk + paraphrase çeşitliliği arttıkça e5-large
   margin avantajı önem kazanır.
3. P2 dersi: "Spec uyarılarını küçümseme — Phi-3 'orta' diyordu, devam
   ettik, kaybettik." Empirical kanıt margin farkını gösterdi → modern
   model risk-yönetimli seçim.
4. Instruction prefix (`query: ` / `passage: `) E5 ailesi native; query
   ve döküman semantiklerini ayrı temsil ediyor → retrieval kalitesi ek
   artırılabilir.

**Risk yönetimi:** `EMBEDDING_MODEL` ENV var ile config-driven; `rag/embeddings.py`
single line switch ile emrecan'a geçilebilir. Production'da latency kritikse
veya küçük korpusta kalırsa fallback hazır.

**Maliyet:** e5-large indirme 2.1 GB HF cache; encode CPU/M-series MPS'te
~0.1 sn/metin (binlerce chunk indeks ~5-15 dk).

**Kanıt dosyaları:**
- `agents/scripts/embedding_benchmark.py` (4 model, kolay test)
- `agents/scripts/embedding_benchmark_hard.py` (2 model, distractor + paraphrase)
- `agents/data/seed_corpus.txt`, `agents/data/seed_corpus_hard.txt`

---

## Açık konular (devam ediyor)

(F-1 ÇÖZÜLDÜ — Sapma 7'ye taşındı.)

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
