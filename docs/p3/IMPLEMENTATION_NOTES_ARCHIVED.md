# P3 — Implementation Notes (Sapmalar)

> Spec'ten bilinçli ayrılan her karar burada gerekçesiyle kayıt altında.
> P1 + P2 pattern'inin devamı (P2'de 31 sapma). Task'lar ilerledikçe ek alır.
> Faz sonu inline callout'larla SPEC.md'ye taşınır + bu dosya
> `IMPLEMENTATION_NOTES_ARCHIVED.md` olur.
>
> **As-of:** 2026-05-03 (P3 FINALIZED — 39 sapma)

---

## Task 4 perf optimizasyonları (post-finalize iyileştirmeler)

### Sapma 36 — TurkishEmbedder LRU query cache + Qdrant compat suppression

**Durum (perf):** Task 5 P1 entegrasyonu sonrası `/ask/v2` 40 sn ölçüldü;
embedder cold start lifespan ile çözüldü ama her query için 30-100 ms
encode kaldı. Tekrarlayan sorularda cache faydalı.

**Uygulanan (cache):** `TurkishEmbedder.__init__` `query_cache_size=128`
parametresi + `OrderedDict`-bazlı manual LRU. Cache key prefixed text
(E5 prefix dahil); `embed_query` lookup → hit'te encode skip + LRU
move-to-end; miss'te encode + en eskisi evict. Telemetry: `cache_stats`
property (hits/misses/size/capacity). 0 → cache devre dışı.

**Uygulanan (Qdrant compat):** `Sapma 9 fix` — `QdrantClient(...,
check_compatibility=False)` Indexer + Retriever'da. Client 1.17 vs server
v1.12.4 minor mismatch uyarısını sustur. Smoke + 22 testte uyumsuzluk
gözlenmedi; server upgrade bilinçli karar olur (Sapma 1 sabit pin
felsefesi).

**Empirical:** 5 cache test (test_embed_cache.py) — hit skip, miss encode,
LRU eviction, move-to-end ordering, disabled mode. Tümü PASSED.

### Sapma 37 — CrewAI post-validator (Sapma 24 follow-up, defense in depth)

**Durum:** Sapma 24'te Writer hallucination prompt-level fix ile çözüldü
ama production güvencesi için ek katman değerli.

**Uygulanan:** `agents/crew/validators.py` — `validate_writer_output(answer,
allowed_sources) -> ValidationResult`. İki seviye:
1. **Akademik referans regex:** `r"\b[A-ZÇĞ...]+(?:,\s+[A-Z]\.)?\s+\(\d{4}\)"`
   — "Newton, I. (1687)" pattern'i flag'ler.
2. **Source whitelist:** "Kaynaklar" başlığı altındaki dosya adları
   `allowed_sources` set'inde değilse → `unknown_sources`.
Hard fail değil; `is_clean: bool` + warnings listesi. test_crew.py'da
crew kickoff sonrası çağrılır.

**Empirical:** 6 unit test (test_validators.py) — clean/academic/unknown/
no-section/bold/empty case'leri. Tümü PASSED.

### Sapma 38 — `/ask/v2/stream` SSE endpoint (node-level streaming)

**Durum:** `/ask/v2` 40 sn → kullanıcı ekran boş bekler. Streaming ile
"ilk node sonu 1-2 sn'de" deneyimi.

**Uygulanan:** `services/api/app/routers/questions.py`'a `/ask/v2/stream`
endpoint. `pipeline.astream()` ile node-level event yayını; her event
SSE format'ta (`event: node_completed\ndata: {...}\n\n`). Final event
cevap + sources + confidence + processing_time_ms.

**Document serialization:** LangChain `Document` JSON-native değil →
`_serialize_event_value` `page_content + metadata` dict'e çevirir.
`retrieved_docs` payload'ı çok büyük → frontend'a sadece özet
(`{count, preview}`).

**Token-level streaming (Anthropic SDK stream):** P3 sonrası iş paketi
— LangGraph state'iyle iç içe daha kompleks; MVP node-level yeter.

---

## Task 5+ Eval A/B Test (post-finalize empirical bulgular)

### Sapma 39 — RAG-with vs baseline A/B sonucu: **B kötü, mimari sağlam**

**Setup:**
- Eval set: P2'nin `ml/data/processed/eval.jsonl` (30 random sample, seed=42)
- Train corpus: P2 `train.jsonl` (348 Q&A) → Qdrant `eduai_eval_corpus`
  collection'ı (her Q&A tek chunk, "Soru: X. Cevap: Y." formatında)
- A (baseline): claude-haiku-4-5 doğrudan, RAG yok
- B (RAG): LangGraph pipeline (retrieve → generate → validate → format)
- Metrikler: ROUGE-1, ROUGE-L, BERTScore F1 (P2 ile aynı)

**Sonuç (30 sample):**

| Metric | A (baseline) | B (RAG) | Δ (B−A) |
|---|---:|---:|---:|
| ROUGE-1 F | 0.3753 | 0.3342 | **−0.041** |
| ROUGE-L F | 0.2257 | 0.1922 | **−0.033** |
| BERTScore F1 | 0.5285 | 0.5178 | **−0.011** |
| Latency (sn) | 6.1 | 10.2 | +4.1 |

- B win-rate ROUGE-1: **8/30 (%27)** — yani %73 baseline daha iyi
- B win-rate BERTScore: **11/30 (%37)**
- B avg confidence: 0.85 (yüksek!) — yanlış cevap için yüksek skor

**Yorum (P2 disiplininin tezahürü):**

P2'de "loss düşüşü ≠ kalite iyileşmesi" dersi: top-1 retrieval skoru ≠
cevap doğruluğu. B avg confidence 0.85 ama B win-rate %27.

Pattern (CSV `agents/data/eval_ab_results.csv` analizinden):
- B'nin **en kötü** olduğu örneklerde (Δ_BERT −0.15 → −0.04): retriever
  **yanlış** chunk getiriyor (eval konusu train'de farklı paragraf'a
  yakın). LLM "Maalesef sağlanan bağlam içerisinde X hakkında bilgi
  bulunmamaktadır" konservatif cevap veriyor; A doğal bilgisinden güzel
  cevap üretiyor.
- B'nin **en iyi** olduğu örneklerde (Δ +0.05): A ve B yarışıyor; gerçek
  RAG faydası küçük.

**4 tanı:**

1. **Yanlış corpus stratejisi:** train.jsonl Q&A pair'leri eval konularını
   tam coverage etmiyor. Retriever **yakın ama yanlış** chunk getiriyor →
   LLM yanlış paragrafa fixate olup kendi doğru bilgisinden uzaklaşıyor.

2. **Validator weak indicator listesi yetersiz:**
   `_WEAK_INDICATORS` listesi `"yeterli bilgi yok"`, `"bağlamda yer almıyor"`
   içeriyor ama eval'da gözlemlenen kalıp **"Maalesef bağlamda... bilgi
   bulunmamaktadır"**, **"kaynaklar yetersiz"** — bu cevaplar weak
   yakalanmıyor → retry tetiklenmiyor (avg attempts 1.23). Liste
   genişletilmeli.

3. **Baseline güçlü:** claude-haiku-4-5 Türkçe lise sorularında zaten
   yetkin (Wikipedia + general training). RAG context ek bilgi sağlamak
   yerine **noise** ekliyor — model kendi bilgisinden uzaklaşıyor.

4. **Recall@k metriği yok:** "Doğru chunk top-4'te var mı?" ölçmedik.
   P3'ün gerçek faydası **retrieved context cevapla alakalı** olduğunda
   kanıtlanır. CONCEPT.md örneği ("tarih_9_unite3.pdf") gibi gerçek
   ders kitabı PDF'leri ile retest gerek.

**Mimariden çıkan ders:**

> **RAG mimarisi sağlam, sorun corpus + validator kalibrasyonu.**

Smoke test'te (Task 1-5) RAG mükemmel çalıştı (top score 0.90, doğru
cevap). Bu A/B test sonucu **mimariyi geçersizleştirmez**, **corpus
stratejisini sorgular**:
- Q&A pair'leri RAG corpus için zayıf (chunk = full Q&A → retrieval
  benzer ama yanlış sorulara fixate)
- Düz ders kitabı paragrafları (CONCEPT örneği) → retrieval konuya bağlı
  bilgi getirir, LLM context'ten cite eder

**Production önerileri (P3 sonrası iş paketleri):**

1. **Validator weak indicator listesini genişlet** — "Maalesef", "bilgi
   bulunmamaktadır", "yetersiz", "veri yok" eklensin (Sapma 39 follow-up).
2. **Real ders kitabı corpus'u** ile retest — MEB müfredat PDF'leri vs.
3. **Recall@k retrieval metriği** — `agents/scripts/eval_retrieval.py`
   yeni script (her eval Q için doğru chunk top-k'de mi).
4. **CrewAI route** — basit sorularda LangGraph yetersizse (B avg %27
   win), karmaşık çok-disiplinli sorularda CrewAI'a yönlendir; basit
   sorularda **baseline doğrudan** kullan (corpus iyi değilse yarar yok).

**Empirical kanıt:** `agents/data/eval_ab_results.csv` 30 satır + manuel
rating slot'ları (kullanıcı 1-5 puanlamak için).

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

### Sapma 9 — qdrant-client 1.17.1 vs server 1.12.4 minor version mismatch

**Durum:** `pip install qdrant-client>=1.12` floor pin'i 1.17.1 çekti;
docker-compose.yml Sapma 1'de `qdrant/qdrant:v1.12.4` sabit pin'lendi.
Client ilk request'te uyarı atar:
> `Qdrant client version 1.17.1 is incompatible with server version 1.12.4.
> Major versions should match and minor version difference must not exceed 1.`

**Şu an etki:** Yok — index_file + scroll + create_collection + payload_index
işlemleri tüm çalıştı, smoke test PASSED. Qdrant minor version'ları
genellikle wire-protocol seviyesinde geriye uyumlu.

**Karar:** Şimdilik uyarı korunuyor (development sinyali olarak); susturmuyoruz.
Üç fix path mevcut:
1. **Server upgrade** (`qdrant/qdrant:v1.13.x` veya v1.17.x): docker-compose.yml
   güncellenir; Qdrant 1.13+ collection schema değişikliği yok.
2. **Client pin** (`qdrant-client>=1.12,<1.13`): pip floor'u kısıtla; sentence-transformers
   ile transitive uyumluluk denenmemiş.
3. **`check_compatibility=False`** indexer constructor'ında: uyarı susar
   ama latent bug'lar gizlenir.

Task 5 sırasında P1 API container'ı qdrant-client çekecek; o zaman compatibility
yeniden değerlendirilir.

### Sapma 10 — `langchain-text-splitters` Türkçe-uyumlu separator'lar

**Spec:** `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)`
**Uygulanan:** Default separator listesi yerine `["\n\n", "\n", ". ", "? ",
"! ", "; ", " ", ""]`.
**Gerekçe:** LangChain default İngilizce odaklı (`["\n\n", "\n", " ", ""]`);
Türkçe metinde cümle sonu noktalama (".", "?", "!") agresif chunk sınırı olur.
Smoke test'te 5 chunk doğru cümle sınırlarında bölündü → metadata page_num=1
korundu, en kısa chunk 80+ char (mantıklı).

### Sapma 11 — chunk_size karakter-bazlı (token değil)

**Spec:** `chunk_size=500` belirtilmiş, birim açıkça yazılmamış.
**Uygulanan:** Karakter bazlı (`length_function=len` default).
**Gerekçe:** Token-aware splitting embedder tokenizer'ına bağımlılık ekler
(import zamanı +tokenizer load). Karakter bazlı 500 ≈ 120 Türkçe token,
e5-large 512 token max sequence içinde rahatça sığar. İleride yetersiz
gelirse `RecursiveCharacterTextSplitter.from_huggingface_tokenizer(...)`
ile token-aware'e geçilir.

## Task 6 — Testler

### Sapma 33 — Indexer/Retriever `client` DI parametresi

**Spec:** `DocumentIndexer(qdrant_url, collection_name, embedder)` ve
`EduRetriever(qdrant_url, collection_name, embedder)` — client her __init__'te
URL'den yaratılıyordu.
**Uygulanan:** Her ikisine `client: QdrantClient | None = None` opsiyonel
parametre eklendi. Default None → URL'den yaratılır (geriye uyumlu);
test'te `QdrantClient(":memory:")` geçirilir.
**Gerekçe:** Test'te in-memory client kullanmak için DI şart. monkey-patch
veya `__new__` + manuel attr set gibi kırılgan workaround'lardan kaçınmak
için temiz seam (DI > monkey-patching). Production kodu davranışı aynı
(None default).

### Sapma 34 — FakeEmbedder + MockLLM (gerçek model yok)

**Durum:** Test'lerde gerçek e5-large (2 GB) yüklemek + Anthropic API
çağırmak:
- CI'da süre + token cost
- HF cache yan etki test izolasyonu kırar
- Network bağımlılığı flake risk

**Uygulanan:**
- `FakeEmbedder`: keyword-bazlı 16-dim L2-normalize vektör.
  TurkishEmbedder protokolüne uyar (`embed_documents`, `embed_query`,
  `vector_size`, `model_id`).
- `MockLLM`: `LLMBackend` Protocol uygulaması; `responses` listesinden
  sırayla döndürür; tükenince son cevabı tekrar (sonsuz retry test'i için).

**Test felsefesi:** Bu testler **akış doğruluğu**na odaklı (DI bağlanıyor mu,
filter çalışıyor mu, retry logic doğru mu). **Semantik kalite** ayrı:
gerçek e5-large smoke test'lerinde empirical doğrulandı (Task 2 retriever
score 0.90, Task 5 confidence 0.89).

### Sapma 35 — Test arası retriever singleton reset (autouse fixture)

**Durum:** `agents.graph.nodes._retriever_singleton` module-level cache
(Sapma 30 fix); bir test'te set edilirse sonraki testlere sızar.
**Uygulanan:** `conftest.py`'da `reset_retriever_singleton` autouse
fixture — her test başında ve sonunda None'a sıfırlar.
**Gerekçe:** Fixture isolation kuralı. Pipeline testleri singleton'ı
populated retriever'a set ederken RAG testlerinin singleton'a dokunması
gerekmiyor → test arası temiz state şart.

### pytest.ini (yeni)

`agents/pytest.ini` eklendi:
- `asyncio_mode = strict` → async testler `@pytest.mark.asyncio` mark
  gerektirir (intent açık).
- `testpaths = tests` → discovery scope.
- `addopts = -ra --strict-markers` → unknown marker fail-fast.

### Test sonuçları (0.18 sn / 11 test)

| Test | Açıklama |
|------|----------|
| test_index_and_retrieve | TASKS Task 6.1 — küçük metin yükle → retrieve sonuç var |
| test_retrieve_with_subject_filter | TASKS Task 6.2 — subject filtresi |
| test_empty_retrieve | TASKS Task 6.3 — boş collection → boş liste |
| test_duplicate_skip (ek) | Sapma 32 fix testi — aynı dosya → 0 chunk |
| test_source_name_override (ek) | Sapma 32 — source_name doc_id'yi etkiler |
| test_retrieve_k_parameter[1/2/4] (ek, parametrize) | k=N sınırı |
| test_full_pipeline | TASKS Task 6.4 — soru → cevap + sources |
| test_retry_logic | TASKS Task 6.5 — kısa cevap → retry tetikleniyor |
| test_pipeline_max_attempts | TASKS Task 6.6 — 3 deneme zorla bitiyor |

Warning (in-memory Qdrant): "Payload indexes have no effect in the local
Qdrant" — production server'da etkili, test'te no-op. Etki yok.

---

## Task 5 — P1 API entegrasyonu

### Sapma 26 — Monorepo PYTHONPATH cascade

**Spec:** TASKS.md "Monorepo yapısı için PYTHONPATH'e dikkat et" — strateji
belirsiz.
**Uygulanan:** Lokal dev başlangıcı:
```bash
cd services/api
PYTHONPATH=$(pwd):$(pwd)/../.. uvicorn app.main:app --port 8000
```
Yani PYTHONPATH'a iki yol: `services/api` (P1 `app.X` import'ları için)
+ repo root (`agents.X` import'ları için).
**Gerekçe:** P1 mevcut `from app.routers import documents` pattern'ini
kırmadan agents/ entegrasyonunu sağlamak. Production Docker'da Dockerfile
WORKDIR=/app/services/api + COPY agents/ /app/agents ile aynı etki.

### Sapma 27 — Shared `.venv-agents` venv (services/api ayrı kurulum yok)

**Durum:** services/api/requirements.txt sadece fastapi/uvicorn/pydantic-
settings/structlog/python-multipart. agents/requirements.txt LangChain +
LangGraph + CrewAI + sentence-transformers + torch + ...
**Uygulanan:** Tek venv `.venv-agents`'a her ikisi yüklendi:
```bash
pip install -r agents/requirements.txt
pip install -r services/api/requirements.txt
```
**Gerekçe:** Lokal dev için iki ayrı venv yönetmek pragmatik değil; aynı
Python süreci her ikisini kullanıyor. Production Docker'da services/api
container'ı her iki requirements'ı yükler.

### Sapma 28 — `DocumentUploadResponse.chunks_indexed` field

**Spec:** SPEC.md DocumentUploadResponse şemasında bu alan yoktu.
**Uygulanan:** `chunks_indexed: int = Field(default=0, ge=0)`.
**Gerekçe:** TASKS.md ekstra gereksinimi: "Response'a 'chunks_indexed'
sayısını ekle". Default=0: P1 endpoint'lerini geriye uyumlu kılar
(eski response model'leri 0 alır), duplicate-skip senaryosunda da 0
döner ve "yeni chunk yüklenmedi" anlamı net.

### Sapma 29 — Tempfile pattern (UploadFile → indexer.index_file)

**Durum:** `DocumentIndexer.index_file()` `Path` bekliyor; FastAPI
`UploadFile` stream/bytes API'si.
**Uygulanan:** `tempfile.NamedTemporaryFile(suffix=ext, delete=False)`
ile içeriği geçici diske yaz, indexer'a path geçir, `try/finally`'de
sil.
**Gerekçe:** indexer'ı UploadFile-aware yapmak abstraction sızıntısı
(indexer FastAPI'den bağımsız kalmalı); tempfile pattern kütüphane
seamlerini koruyor.

### Sapma 30 — Lifespan eager initialization (Sapma 21 ÇÖZÜLDÜ)

**Durum:** Sapma 21'de "her node call'da yeni EduRetriever() → ilk request
~30 sn cold start" flag'lenmişti.
**Uygulanan:**
1. `agents/graph/nodes.py`: module-level `_retriever_singleton` + `_get_retriever()`
   helper. Birinci çağrıda yaratılır, sonrakiler paylaşır.
2. `services/api/app/main.py` lifespan: startup'ta DocumentIndexer +
   pipeline + `_get_retriever()` eager init. Vector_size erişimi embedder'ı
   tetikler → e5-large model startup'ta yüklenir.
**Etki:** İlk `/ask/v2` request artık cold start cezası yaşamaz (modeller
zaten warm). Smoke test'te ilk request ~40 sn ama bu Anthropic API call
(~10 sn) + LangGraph overhead, embedder load değil.

### Sapma 31 — `.env` cascade (services/api/main.py)

**Durum:** pydantic-settings `.env` dosyasını sadece cwd'den okur. cwd
`services/api` → ANTHROPIC_API_KEY orada yok (ml/.env'de).
**Uygulanan:** `_load_env_cascade()` Settings init'inden ÖNCE çalışır;
agents/.env → repo_root/.env → ml/.env sırasıyla `load_dotenv` çağırır
(noqa: E402 import order).
**Gerekçe:** agents/graph/pipeline.py ile aynı pattern → tek bir .env
dosyasında ANTHROPIC_API_KEY tutmak yeterli; her modül otomatik bulur.

### Sapma 32 — `index_file(source_name=...)` parametresi

**Durum:** API tempfile path'i (`/tmp/tmp9j8k2l.txt`) `_compute_doc_id`'e
verildiğinde `tmp9j8k2l_<hash>` doc_id üretiyor → her upload yeni doc_id
→ duplicate-skip çalışmaz.
**Uygulanan:** `DocumentIndexer.index_file(file_path, metadata, source_name=None)`
ek parametre. `source_name` verildiğinde doc_id stem'i bundan hesaplanır
(örn. "tarih_tanzimat.txt"). Default=None lokal script senaryosunda
geriye uyumlu (file_path.name fallback).
**Empirical doğrulama:** Aynı dosya upload edildi → `chunks_indexed: 0`
(duplicate-skip OK). İlk upload `chunks_indexed: 5` (yeni indeks).

---

## Task 5 Smoke Test Sonuçları

`POST /v1/documents/upload` (tarih_tanzimat.txt, ikinci yükleme):
```json
{"chunks_indexed": 0, "status": "ready", ...}  // duplicate-skip ✓
```

`POST /v1/questions/ask/v2` (Tanzimat ne zaman + neler getirdi?):
```json
{
  "answer": "# Tanzimat Fermanı: İlan Tarihi ve Getirdikleri\n\n## ...",
  "confidence": 0.8962,
  "sources": ["tarih_tanzimat.txt"],
  "processing_time_ms": 40869
}
```
- Markdown başlıklı, [1]-[4] kaynak referanslı
- 40.8 sn (~10 sn Anthropic + LangGraph overhead; embedder zaten warm)
- needs_retry tetiklenmedi (validator OK)

---

## Task 4 — CrewAI multi-agent

### Sapma 22 — F-2 ÇÖZÜLDÜ: CrewAI `>=0.80.0` → `1.14.3` (major bump)

**Spec:** `crewai>=0.80.0` (2024 yazımı).
**Uygulanan:** `crewai==1.14.3` yüklü; floor pin koruyor (>=0.80) ama
gerçekte 1.x major.
**API değişiklikleri:** Çok minimal — Pydantic-driven kwargs uyumlu kaldı:
- `Agent(role=..., goal=..., backstory=..., tools=[...], llm=..., verbose=True)`
- `Task(description=..., expected_output=..., agent=..., context=[...])`
- `Crew(agents=..., tasks=..., process=Process.sequential, verbose=True)`
- `from crewai.tools import tool, BaseTool` decorator pattern aynı.

**Tek değişen:** LLM provider config. CrewAI 1.x **LiteLLM wrapper'ı** içeri
gömüldü → `from crewai import LLM` ile kullanılıyor. Model string'i provider
prefix'li: `anthropic/claude-haiku-4-5`. ANTHROPIC_API_KEY ENV otomatik
çekilir.

**requirements.txt güncellemesi gerek mi?** Hayır — `crewai>=0.80.0` floor
pin'i 1.14.3'ü zaten çekiyor. Ek pin gerek değil; 2.x'e major bump'a
karşı `<2` cap eklenebilir gelecekte.

### Sapma 23 — LLM provider prefix `anthropic/<model>` (LiteLLM)

**Durum:** CrewAI Agent'lara LLM atanırken `LLM(model="claude-haiku-4-5")`
denemesi LiteLLM auto-detect'i provider'ı bulamayabilir.
**Uygulanan:** `LLM(model="anthropic/claude-haiku-4-5")` — explicit prefix.
**Gerekçe:** LiteLLM 100+ provider destekliyor; explicit prefix ambiguity
önler ve production'da daha güvenilir. Sapma 16 (graph/llm.py'da Anthropic
SDK direkt kullanılıyor) ile farklı: LangGraph node'ları AsyncAnthropic ile,
CrewAI agent'ları LiteLLM ile çağırıyor → iki ayrı çağrı yolu, aynı API
key paylaşılıyor.

### Sapma 24 — Writer "Kaynaklar" hallucination — ÇÖZÜLDÜ (prompt fix)

**Gözlem (ilk smoke):** Writer cevabın sonuna **bağlamda olmayan** kitap
referansları ekledi:
> "- Newton, I. (1687). *Principia Mathematica* – Hareket Yasaları"
> "- Osmanlı Arşivi. *Tanzimat Reformları ve Kurumsal Dönüşüm*"

**Kök neden:** Researcher → Writer arası context **özet** olarak akıyor
(raw chunk değil); Writer "akademik kaynak listesi" boşluğunu uydurarak
doldurdu. Backstory "10 yıllık öğretmen" personası bu refleksi besliyordu.

**Uygulanan fix (iki katmanlı):**
1. **Writing task description** (`agents/crew/tasks.py`): "KAYNAKLAR bölümü
   kuralları (KRİTİK)" başlığı + 3 explicit kural:
   - Sadece Researcher çıktısında `[kaynak: <dosya>.txt]` formatındaki
     dosya adlarını kullan.
   - Kitap/yazar/tarih/yayınevi/ISBN/dergi/link EKLEME.
   - Akademik referans formatı (APA/MLA) kullanma.
2. **Writer agent backstory** (`agents/crew/agents.py`): "UYDURMA YAYIN
   BİLGİSİ yazmazsın — sadece sana gerçekten verilen dosya adlarını
   referans gösterirsin."

**Re-test sonucu:** Yeni "Kaynaklar" çıktısı:
```
- fizik_newton.txt: Newton'un hareket yasaları (I., II., III. yasalar)
- tarih_tanzimat.txt: Osmanlı'nın Tanzimat dönemi reformları ve modernleşme süreci
```
Sıfır uydurma yayın bilgisi; sadece dosya adı + içerik etiketi (talep
edilen format).

**Geriye kalan risk:** Bu prompt-level fix; LLM'in bambaşka bir bağlamda
benzer halüsine yapma olasılığı tamamen sıfırlanmadı. Production'da
**post-validator** (LangGraph validate_node benzeri, CrewAI çıktısına
uygulanan kalite kontrol) eklenebilir — Task 5/6 sonrası eval set
empirik olarak gösterirse.

### Sapma 25 — Token cost gözlemi (8827 token / 3 LLM call)

**Durum:** Smoke test 8827 toplam token (~$0.04 claude-haiku-4-5'te).
3 LLM call: Researcher tool reasoning + Writer reasoning + final.

**Etki:**
- LangGraph pipeline (Task 3): tek LLM call, ~3000 token, ~$0.013
- CrewAI crew (Task 4): 3 LLM call, ~8800 token, ~$0.04 — **3x maliyet**

**Karar:** CrewAI sadece **karmaşık (multi-disciplinary) sorular** için
kullanılmalı (Task 5'te endpoint routing); basit sorularda LangGraph yeter.
Bu zaten CONCEPT.md'nin önerdiği mimari ("LangGraph ana, CrewAI karmaşık
sorular"). Maliyet bilinci eklendi.

### Smoke test sonucu

`PYTHONPATH=. python -m agents.crew.test_crew`:
- Soru: "Newton'un hareket yasaları ve Osmanlı'nın modernleşme süreci
  arasında benzer bir dinamik var mı? Açıkla."
- Researcher iki ayrı RAG çağrısı yaptı (fizik + tarih subject filter)
- Writer F=m·a → Tanzimat dış kuvvet/kütle/ivme analojisi kurdu
- Eylemsizlik prensibi → toplumsal direniş bağlantısı
- Markdown formatı, başlıklar, listeler tam
- Token: 8827 total (3 successful_requests)

---

## Task 3 — LangGraph pipeline

### Sapma 16 — Ek `agents/graph/llm.py` modülü (SPEC'te yok)

**Spec:** `graph/` klasörü 4 dosya öneriyor: state, nodes, edges, pipeline.
**Uygulanan:** 5 dosya — yeni `llm.py` eklendi.
**Gerekçe:** TASKS.md `LLM_BACKEND` ENV-driven backend swap istiyor
(anthropic/qwen3-local/vllm). Bunu nodes.py içine inline koymak iki sorun:
1. nodes.py uzun olur, SRP ihlali (her node hem prompt hem provider seçer).
2. Task 4 CrewAI tool'u aynı LLM'i çağıracak — ortak factory gerek.
`get_llm()` factory + `LLMBackend` Protocol pattern → temiz seam.

### Sapma 17 — Qwen3LocalBackend + VLLMBackend stub'ları

**Spec:** TASKS.md "P2 model integration Task 3 sonu veya 5'te" diyor.
**Uygulanan:** `Qwen3LocalBackend.__init__` ve `VLLMBackend.__init__` →
`NotImplementedError` (mesajlı stub).
**Gerekçe:** Task 3 hedefi pipeline'ı ayağa kaldırmak; P2 modeli **macOS'ta
çalışamaz** (bitsandbytes Linux/CUDA-only). Stub'lar API yüzeyini
kilitliyor (Protocol uyumlu) ama implementasyon Task 5+'da Linux/Colab
path'inde yapılacak. Hata mesajı kullanıcıya açık talimat veriyor.

### Sapma 18 — Validator MVP heuristic (length + weak indicators)

**Spec:** `len(state["answer"]) < 50 and state["attempts"] < 3` örneği.
**Uygulanan:** Length kontrolü + Türkçe belirsizlik kalıpları
(`"bilmiyorum"`, `"yeterli bilgi yok"`, `"emin değil"`, `"bağlamda yer
almıyor"`).
**Gerekçe:** SPEC örneği sadece length'e bakıyordu — model uzun ama
"bilmiyorum" diyebilir; bu retry tetiklemesi gerek. P2 dersi: "kalite
metrikleri sample inspection'la birlikte." Bu MVP; Task 4/5'te
NLI/LLM-as-judge ile değiştirilecek (Sapma 22 plan).

### Sapma 19 — SPEC `await retriever.retrieve(...)` sync çağrı

**Spec:** `nodes.py` örneğinde `await retriever.retrieve(...)`.
**Uygulanan:** `retriever.retrieve(...)` (sync; Sapma 12 ile tutarlı).
**Gerekçe:** retriever sync (qdrant-client blocking); LangGraph node
async olsa da içinde sync IO çağırmak meşru. SPEC örneği yanıltıcı.

### Sapma 20 — Confidence = top retrieved doc score

**Spec:** "Basit heuristic: cevap uzunluğu, belirsiz kelimeler vb."
**Uygulanan:** `confidence = float(docs[0].metadata["score"])` —
retriever'ın cosine sim'i.
**Gerekçe:** Cevap uzunluğu yanıltıcı (uzun cevap zayıf olabilir).
Top retrieved doc skoru retrieval kalitesini ölçer; düşükse model
zaten az bağlamla cevap üretiyor demektir → güven düşük. E5-large
benchmark avg 0.86; bu metrik anlamlı dağılım veriyor (smoke test'te
0.89). LLM-as-judge ile değiştirilebilir.

### Sapma 21 — Her node call'da yeni `EduRetriever()` (TODO Task 5)

**Durum:** `retrieve_node` her çağrıda `EduRetriever()` oluşturuyor →
embedder lazy-load ilk request'te 33 sn cold start.
**Etki:** Smoke test toplam 49 sn (33 sn embedder + 16 sn diğer).
Production'da kabul edilemez.
**Plan:** Task 5'te FastAPI lifespan'inde global embedder + retriever
singleton oluştur, dependency injection ile node'lara geçir. MVP'de
şimdilik kabul.

### Smoke test sonucu

`python -m agents.graph.pipeline` (Tanzimat sorusu, subject=tarih):
- retrieve: 39.857 sn (cold embedder)
- generate: 9.024 sn (claude-haiku-4-5)
- validate: 0.003 sn
- format: 0.000 sn
- **Cevap kalitesi:** markdown, 4 kaynak atıfı `[1]-[4]`, Türkçe pedagojik
- **Confidence:** 0.8914 (top retrieved doc), needs_retry=False, attempts=1
- **Sources:** `['tarih_tanzimat.txt']`

---

## Task 2 — RAG retriever

### Sapma 12 — `retrieve()` sync API (SPEC örnekte `await ... .retrieve(...)`)

**Spec:** `nodes.py` örneğinde `await retriever.retrieve(...)` yazıyor.
**Uygulanan:** `EduRetriever.retrieve()` sync.
**Gerekçe:** `qdrant-client` blocking API kullanıyor; `AsyncQdrantClient` ayrı
ama agents/ tek-flag stratejisinde gerek yok. LangGraph node'u (Task 3) `async def`
olabilir ve içinde sync fonksiyon çağırabilir → API basit, type ergonomi yüksek.
Eğer Task 3'te node throughput sorunu çıkarsa `AsyncQdrantClient`'a geçiş
tek `__init__` değişikliği.

### Sapma 13 — `Document` tipi: `langchain_core.documents.Document`

**Spec:** Sadece "Document" diyor, tip spesifik değil.
**Uygulanan:** `langchain_core.documents.Document`.
**Gerekçe:** LangGraph retrieve_node (Task 3) ve CrewAI tool (Task 4) zaten
LangChain ekosistemi → ortak tip integration kolaylığı; ek wrapper sınıfı
yaratmak gereksiz abstraction olur (P2 dersi: "premature abstraction yok").

### Sapma 14 — Score `metadata["score"]`'da saklanıyor

**Spec:** Dönüş tipi `list[Document]`, score yok.
**Uygulanan:** Cosine similarity skoru `metadata["score"]` altında.
**Gerekçe:** Task 3 validator node'u "score < threshold → retry" pattern'i
ile çalışacak (LangGraph conditional edges); skor erişimi şart. Tuple
`(Document, float)` döndürmek SPEC imzasını bozar; metadata'da saklamak
hem geriye uyumlu hem de tüketici tarafta `doc.metadata["score"]` ergonomik.

### Sapma 15 — `get_context_string` formatı: numaralı + Türkçe header

**Spec:** Sadece "chunk'ları tek string'e birleştir" diyor, format belirsiz.
**Uygulanan:**
```
[1] (kaynak: tarih_tanzimat.txt, sayfa: 0)
<chunk metni>

[2] (kaynak: ...)
...
```

**Gerekçe:**
1. **Numaralandırma** model'in alıntı yapmasını kolaylaştırır (`[1]'de
   söylendiği gibi...`); pedagojik tonda doğal.
2. **Kaynak + sayfa header** validator/UI'da source listesi çıkarmak için
   parse edilebilir (Task 3 format_node bu metadata'yı zaten state'ten
   okuyacak ama context string'in kendisi de erişilebilir kalsın).
3. Türkçe (kaynak, sayfa) → P1+P2 disiplini, model'in Türkçe ton'unu
   bozmaz.

**Empirical doğrulama:** Test "Tanzimat Fermanı ne zaman çıktı?" sorusunda
top-1 score=0.9044 (E5-large benchmark avg 0.86'nın üzerinde) → tutarlı
yüksek kalite. Subject filter (tarih) tüm 4 sonucu döndürdü, fizik filter
0 sonuç → filter syntax doğru.

---

### Sapma 8 — `torch==2.9.1` sabit pin → `torch>=2.9` floor pin

**Spec/önceki karar:** agents/requirements.txt'te P2 ml/ pattern'inden
`torch==2.9.1` sabit pin devralındı.
**Uygulanan:** `torch>=2.9` floor pin.
**Gerekçe:** agents/ venv'i kurulduğunda sentence-transformers transitive
olarak torch 2.11.0 çekti (sürüm 5.4.1 daha güncel torch tercih ediyor).
Sabit pin'i zorlamak için downgrade gerekirdi — gereksiz çünkü:
1. agents/ macOS dev için sürüm uyumu kritik değil; bitsandbytes platform
   guard ile zaten Linux-only.
2. P2 ml/'in sabit pin gerekçesi (Colab CUDA 12.x + bitsandbytes ABI) burada
   yok; agents/ Anthropic API üzerinden geliştirilecek (Task 3 dev path).
3. Lokal Qwen3 inference Task 5'te entegre olursa, o zaman ml/ venv'inden
   import etmek veya lokal Linux production lock dosyası yazmak daha
   pragmatik.

**Switch path:** Production Linux + CUDA için `agents/requirements.lock.txt`
(pip freeze) eklendiğinde torch sürümü o lock dosyasıyla sabitlenir.

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
