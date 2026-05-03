# EduAI Agents (P3)

RAG + LangGraph state machine + CrewAI multi-agent orchestration. P3 fazının
implementasyonu — P2 fine-tuned Qwen3-4B adapter'ı + Türkçe ders kitabı
chunk'ları + LangGraph pipeline + CrewAI multi-disciplinary çözümlemesi.

> **Status:** P3 implementation tamamlandı (Task 0 → 6, 35 sapma kayıtlı). ·
> **Spec:** [`docs/p3/SPEC.md`](../docs/p3/SPEC.md) (inline callout'larla
> güncellenmiş hâli) · **Karar tarihçesi:**
> [`docs/p3/IMPLEMENTATION_NOTES_ARCHIVED.md`](../docs/p3/IMPLEMENTATION_NOTES_ARCHIVED.md)

---

## 1. Mimari özet

```
Soru → /v1/questions/ask/v2 (FastAPI)
         ↓
LangGraph pipeline (compiled, app.state singleton)
   retrieve → generate → validate → (retry|format) → END
         ↓                ↓                  ↓
  EduRetriever      LLMBackend          format_node
  (Qdrant +         (Anthropic         (sources
   e5-large)         dev / Qwen3        toplama)
                     prod)
```

**Bileşenler:**
- **`rag/`** — `TurkishEmbedder` (e5-large), `DocumentIndexer` (TXT/PDF →
  Qdrant), `EduRetriever` (query → top-k LangChain Document)
- **`graph/`** — `AgentState` TypedDict, 4 async node, conditional edge,
  compiled pipeline. LLM backend abstraction (`anthropic` / `qwen3-local`
  / `vllm` config-driven).
- **`crew/`** — Researcher + Writer agents (CrewAI 1.14), multi-disciplinary
  sorular için.
- **`tests/`** — pytest, in-memory Qdrant + FakeEmbedder + MockLLM (CI hızlı).

---

## 2. Klasör yapısı

```
agents/
├── data/                      Test korpusu (tarih_tanzimat.txt, fizik_newton.txt, seed_corpus*.txt)
├── scripts/                   Tek seferlik araştırma/seed scriptleri
│   ├── embedding_benchmark.py        F-1 4-model karşılaştırma
│   ├── embedding_benchmark_hard.py   F-1 distractor'lı zor test
│   └── index_seed.py                 Tarih + fizik korpusunu Qdrant'a yükle
├── rag/                       embeddings.py · indexer.py · retriever.py
├── graph/                     state.py · llm.py · nodes.py · edges.py · pipeline.py
├── crew/                      tools.py · agents.py · tasks.py · test_crew.py
├── memory/                    (Konuşma geçmişi — Task 4+ memory entegrasyonu)
├── tests/                     conftest.py · test_rag.py · test_pipeline.py
├── test_connection.py         Qdrant smoke (Task 0)
├── test_retrieval.py          Retriever smoke (Task 2)
├── requirements.txt           agents/ bağımlılıkları (floor pin)
├── pytest.ini                 asyncio_mode=strict + testpaths
├── .env.example               LLM_BACKEND, ANTHROPIC_API_KEY, QDRANT_URL
└── README.md
```

---

## 3. Önkoşullar

- **Python 3.13** (pyenv tavsiye edilir)
- **Docker + docker-compose** — Qdrant container için
- **Anthropic API anahtarı** — dev'de claude-haiku-4-5 (TASKS.md Task 3
  rehberi). [console.anthropic.com](https://console.anthropic.com/)'dan alınır
- **macOS dev:** MPS (Apple Silicon) destekli; bitsandbytes Linux/CUDA-only,
  Qwen3 lokal inference Colab/Linux gerek

---

## 4. Kurulum

```bash
# 1) venv (P1/P2 venv'lerinden ayrı tut — bağımlılık karışmasın)
python -m venv .venv-agents
source .venv-agents/bin/activate

# 2) Tüm bağımlılıklar (langchain stack + langgraph + crewai + sentence-transformers + ...)
pip install -r agents/requirements.txt

# Task 5+ için P1 API bağımlılıklarını da aynı venv'e yükle:
pip install -r services/api/requirements.txt
```

> Sapma 27 detay: tek venv stratejisi (lokal dev). Production Docker'da
> services/api container'ı her iki requirements'ı yükler.

---

## 5. ENV ayarları (`.env` cascade)

`agents/.env.example`'i kopyalayıp `.env` yarat:

```bash
cp agents/.env.example agents/.env
$EDITOR agents/.env  # ANTHROPIC_API_KEY'i doldur
```

**Cascade sırası** (`pipeline.py` ve `services/api/app/main.py` aynı pattern'i
kullanır — Sapma 31):
1. `agents/.env` (tercih edilen)
2. `<repo_root>/.env`
3. `ml/.env` (P2'den geriye uyum)

İlki bulunursa onu yükler; yoksa diğerine geçer. **Tek bir .env dosyasında
ANTHROPIC_API_KEY tutmak yeter.**

Ana ayarlar:
- `LLM_BACKEND` — `anthropic` (dev), `qwen3-local` (Linux/CUDA, Task 5+),
  `vllm` (production)
- `ANTHROPIC_API_KEY` — sk-ant-...
- `ANTHROPIC_MODEL` — `claude-haiku-4-5` default
- `QDRANT_URL` — `http://localhost:6333` (Docker network'te `http://qdrant:6333`)
- `QDRANT_COLLECTION` — `eduai_documents` default

---

## 6. Çalıştırma akışı

### 6.1 Qdrant container

```bash
docker-compose up qdrant -d
# veya: docker-compose up qdrant -d api  (API + Qdrant birlikte)

# Web UI:        http://localhost:6333/dashboard
# REST endpoint: http://localhost:6333
```

> Qdrant `v1.12.4` sabit pin (Sapma 1). Reproducibility için `latest` kullanmadık.

### 6.2 Smoke test'ler (sırayla)

```bash
source .venv-agents/bin/activate

# Task 0 — Qdrant bağlantısı
python agents/test_connection.py
# Beklenen: "Smoke test PASSED"

# Task 1 — Test korpusunu indeksle (tarih + fizik)
PYTHONPATH=. python agents/scripts/index_seed.py
# Beklenen: "tarih_tanzimat.txt 5 chunk, fizik_newton.txt 5 chunk"

# Task 2 — Retriever
PYTHONPATH=. python -m agents.test_retrieval
# Beklenen: "Top score 0.90+"

# Task 3 — LangGraph pipeline (Anthropic API call yapar)
PYTHONPATH=. python -m agents.graph.pipeline
# Beklenen: Markdown cevap + sources

# Task 4 — CrewAI multi-disciplinary (Newton + Tanzimat)
PYTHONPATH=. python -m agents.crew.test_crew
# Beklenen: F=m·a → Tanzimat analojisi
```

### 6.3 P1 API'yi başlat (Task 5)

```bash
cd services/api
PYTHONPATH=$(pwd):$(pwd)/../.. uvicorn app.main:app --reload --port 8000
```

Lifespan'de eager init yapılır (~30 sn cold start, e5-large yüklemesi).
Sonraki istekler hızlı.

> Sapma 26 — PYTHONPATH cascade: P1'in `from app.X` ve agents'ın
> `from agents.X` import'ları tek runtime'da çalışsın diye.

### 6.4 Doküman upload (curl)

```bash
curl -X POST http://localhost:8000/v1/documents/upload \
  -F "file=@agents/data/tarih_tanzimat.txt" \
  -F "title=Tanzimat" \
  -F "subject=tarih" \
  -F "grade_level=9"

# Yanıt:
# {
#   "document_id": "...",
#   "chunks_indexed": 5,    # ilk yükleme
#   "status": "ready",
#   ...
# }
# Aynı dosyayı tekrar yükle → "chunks_indexed": 0 (duplicate-skip, Sapma 32)
```

### 6.5 Soru sorma (RAG cevabı)

```bash
curl -X POST http://localhost:8000/v1/questions/ask/v2 \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Tanzimat Fermanı ne zaman ilan edildi?",
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "subject": "tarih",
    "grade_level": 9
  }'

# Yanıt:
# {
#   "answer": "# Tanzimat Fermanı\n\n3 Kasım 1839'da... [1]",
#   "confidence": 0.89,
#   "sources": ["tarih_tanzimat.txt"],
#   "processing_time_ms": 12000,
#   ...
# }
```

Swagger UI: `http://localhost:8000/docs`

---

## 7. Test çalıştırma

```bash
source .venv-agents/bin/activate

# Tüm testler (in-memory Qdrant + mock LLM, ~0.2 sn)
PYTHONPATH=. pytest agents/tests/ -v

# Sadece RAG akışı
PYTHONPATH=. pytest agents/tests/test_rag.py -v

# Sadece pipeline retry logic
PYTHONPATH=. pytest agents/tests/test_pipeline.py::test_retry_logic -v

# Lint + format
ruff check agents/ services/api/
ruff format --check agents/ services/api/
```

> 11 test, ~0.18 sn. Network/disk yok — Sapma 34 (FakeEmbedder + MockLLM).

---

## 8. Mimari kararlar (özet)

P3 boyunca 35 bilinçli sapma kayıtlı. Önemli olanlar:

| # | Karar | Gerekçe |
|---|-------|---------|
| 7 | **Embedding model = `intfloat/multilingual-e5-large`** (SPEC: emrecan) | Hard benchmark'ta confidence margin 0.86 vs 0.68 — validator threshold için kritik |
| 16 | Ek `graph/llm.py` modülü | Backend abstraction (anthropic/qwen3-local/vllm) — nodes.py SRP |
| 22 | **F-2 ÇÖZÜLDÜ** — CrewAI 0.80 → 1.14 | API uyumlu kaldı; tek değişen LiteLLM provider config |
| 24 | Writer hallucination ÇÖZÜLDÜ | Prompt-level fix: explicit "yayın bilgisi yazma" kuralı |
| 30 | Lifespan eager init (Sapma 21 ÇÖZÜLDÜ) | İlk request cold-start cezasız |
| 32 | `index_file(source_name=...)` parametresi | Tempfile path'i doc_id'yi etkilemesin → duplicate-skip çalışsın |
| 33 | Indexer/Retriever `client` DI | Test'te in-memory Qdrant geçirilebilsin |

Tam liste: [`docs/p3/IMPLEMENTATION_NOTES_ARCHIVED.md`](../docs/p3/IMPLEMENTATION_NOTES_ARCHIVED.md).

---

## 9. Troubleshooting

**`ModuleNotFoundError: agents.rag.X`** — `PYTHONPATH=.` eksik. Repo root'tan
çalıştır: `PYTHONPATH=. python ...`

**Qdrant client version warning** (1.17 vs server 1.12) — Sapma 9; şu an
çalışıyor. Server upgrade için `docker-compose.yml` qdrant tag'ini güncelle.

**`/ask/v2` → 503 "Agent pipeline hazır değil"** — Lifespan startup başarısız
oldu (ANTHROPIC_API_KEY yok ya da Qdrant down). uvicorn loglarında
`p3_runtime_init_failed` event'i kontrol et.

**Embedder cold start 30+ sn** — Beklenen, ilk request modeli yüklüyor.
Lifespan pre-warm yapıyor → uvicorn ready logu sonrası requests hızlı.

**`bitsandbytes` hatası macOS'ta** — Linux/CUDA only. `requirements.txt`'te
`sys_platform != 'darwin'` guard var; macOS dev'de skip edilir.

**HF cache disk dolu** — `~/.cache/huggingface/hub/` 5+ GB olabilir
(e5-large + benchmark modelleri). Kullanılmayanları sil:
`rm -rf ~/.cache/huggingface/hub/models--BAAI--bge-m3` vs.

---

## 10. Geliştirme döngüsü disiplini (P1+P2'den taşındı)

- **Smoke test maliyeti düşük, değeri yüksek** — her component (Qdrant,
  embedder, retriever, pipeline, crew) ayrı doğrulandı; full integration
  öncesi.
- **Sapma yaz** — spec'ten ayrılan her karar `IMPLEMENTATION_NOTES.md`'de
  gerekçesiyle. Faz sonu inline callout'larla SPEC'e taşınır.
- **Loss düşüşü ≠ kalite iyileşmesi** (P2) → **Top-1 ≠ kalite** (P3 F-1
  benchmark): tie kırıldığında confidence margin'e bakıldı.
- **CI gate disiplini:** commit öncesi `ruff check + ruff format --check + pytest`.
- **Risk yönetimi:** ENV override'lar (LLM_BACKEND, EMBEDDING_MODEL,
  QDRANT_URL), fallback path'ler, lifespan eager init failure → graceful
  degradation.
