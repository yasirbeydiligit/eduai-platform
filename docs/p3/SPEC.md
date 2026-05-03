# P3 — Akıllı Ajan: Proje Spesifikasyonu

> **📌 Status:** P3 implementation **FINALIZED** (Task 0 → 6, 35 bilinçli sapma).
> Bu dosya inline **📝 Implementation note** callout'larıyla güncellendi —
> her bölümde uygulanan kararlar gerekçeleriyle belirtilmiş.
>
> **Tam karar tarihçesi:** [`IMPLEMENTATION_NOTES_ARCHIVED.md`](IMPLEMENTATION_NOTES_ARCHIVED.md)
> (35 sapma, Task 0-6 sırasıyla).
>
> **As-of:** 2026-05-03

---

## Proje özeti

**Ne:** RAG + LangGraph + CrewAI ile multi-agent soru-cevap sistemi. P2 modelini kullanır, P1 API'si üzerinden servis edilir.
**Klasör:** `eduai-platform/agents/`
**Bağımlılık:** P1 çalışıyor olmalı. P2 modeli hazır olmalı (ya da mock ile başlanabilir).

> **📥 P2 bağımlılık spesifik (2026-04-28 finalize sonrası):**
> - **Base model:** `Qwen/Qwen3-4B-Instruct-2507` (Apache 2.0)
> - **LoRA adapter:** Drive `/content/drive/MyDrive/eduai_qwen3-4b-instruct-2507_ckpt/`
>   veya HF Hub'a push edilmiş alternatif (P3 başlangıçta karar)
> - **Inference profili:** T4 ~22-25 sn/cevap (geliştirme yavaş);
>   **fallback olarak Anthropic API kullanılması TASKS.md Task 3'te öneriliyor**
> - **Adapter yükleme kod örneği:** [`docs/p2/P3_HANDOFF.md`](../p2/P3_HANDOFF.md) § 3
> - **Inference wrapper imzası:** P3_HANDOFF.md § 3'teki `generate_answer(instruction, context, ...)`
>   imzası `generate_node` ile uyumlu — context parametresi RAG'dan gelecek

---

## Klasör yapısı

```
eduai-platform/
└── agents/
    ├── rag/
    │   ├── indexer.py           ← PDF/TXT → chunks → Qdrant
    │   ├── retriever.py         ← soru → ilgili chunks
    │   └── embeddings.py        ← Türkçe embedding modeli
    ├── graph/
    │   ├── state.py             ← AgentState TypedDict
    │   ├── nodes.py             ← her adım bir fonksiyon
    │   ├── edges.py             ← karar mantığı
    │   └── pipeline.py          ← graph'ı birleştir
    ├── crew/
    │   ├── agents.py            ← Researcher, Writer rolleri
    │   ├── tasks.py             ← görev tanımları
    │   └── tools.py             ← RAG tool, calculator tool
    ├── memory/
    │   └── session_memory.py    ← konuşma hafızası
    ├── tests/
    │   ├── test_rag.py
    │   └── test_pipeline.py
    ├── requirements.txt
    └── README.md
```

> **📝 Implementation note (Sapma 3, 16 — gerçek yapı):**
> - **Sapma 3:** Task 0'da boş feature `.py` dosyaları yerine sadece
>   `__init__.py`'ler — boş dosyalar "yazılmış sanılır" gürültüsünü önler.
> - **Sapma 16:** `graph/` klasörüne **`llm.py`** eklendi (LLM backend
>   abstraction).
> - **Ek dosyalar (SPEC'te yoktu, eklendi):**
>   - `data/` — test korpusu (`tarih_tanzimat.txt`, `fizik_newton.txt`,
>     `seed_corpus*.txt`)
>   - `scripts/` — tek seferlik araştırma scriptleri (`embedding_benchmark*.py`,
>     `index_seed.py`)
>   - `test_connection.py`, `test_retrieval.py` — top-level smoke runner'lar
>   - `pytest.ini`, `.env.example`

---

## Bileşen spesifikasyonları

### rag/embeddings.py

```python
"""Türkçe metin embedding modeli"""
# Model: "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
# Alternatif: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# Her iki model de Türkçe'yi iyi işliyor

class TurkishEmbedder:
    def embed_documents(self, texts: list[str]) -> list[list[float]]
    def embed_query(self, query: str) -> list[float]
```

> **📝 Implementation note (Sapma 7 — F-1 ÇÖZÜLDÜ):** SPEC `emrecan/bert-base-
> turkish-cased-mean-nli-stsb-tr` (2021, 768-dim) öneriyordu. İki aşamalı
> empirical benchmark sonucu (`agents/scripts/embedding_benchmark*.py`)
> **`intfloat/multilingual-e5-large` (1024-dim)** seçildi. Hard test'te her
> ikisi %100 top-1 ama confidence margin farkı belirleyici: e5-large avg
> 0.86 / min 0.83, emrecan avg 0.68 / min 0.58 (yakın distractor pair'lerde
> kayıp riski). Validator threshold-based retry (Task 3) için margin kritik.
> `EMBEDDING_MODEL` ENV var ile config-driven swap kolay; bge-m3 latency
> 65sn'den, gte-multilingual ABI hatasından elendi.

> **📝 Implementation note (ek özellikler):** TurkishEmbedder ek olarak:
> (a) lazy load — `__init__` ucuz, model ilk encode'da yüklenir; (b)
> instruction prefix auto-detect — E5 ailesi için `query: ` / `passage: `
> otomatik (retrieval kalitesini ek artırır); (c) device auto — MPS > CUDA > CPU.

### rag/indexer.py

```python
"""Dökümanları Qdrant'a yükler"""
class DocumentIndexer:
    def __init__(self, qdrant_url, collection_name):
        ...
    
    def index_file(self, file_path: str, metadata: dict) -> int:
        """
        1. PDF/TXT oku
        2. RecursiveCharacterTextSplitter ile chunk (size=500, overlap=50)
        3. Her chunk'ı embed et
        4. Qdrant'a yükle
        Döndür: yüklenen chunk sayısı
        """
    
    def list_documents(self) -> list[dict]:
        """Yüklenen dökümanların listesi + metadata"""
```

> **📝 Implementation note (Sapma 10-11):** Splitter Türkçe-uyumlu
> separator'larla (`["\n\n", "\n", ". ", "? ", "! ", "; "]`) — LangChain
> default İngilizce odaklıydı. `chunk_size=500` **karakter-bazlı**
> (~120 Türkçe token); token-aware splitting tokenizer bağımlılığı ekler,
> MVP'de yeterli.

> **📝 Implementation note (Sapma 32 — source_name parametresi):**
> Imza genişletildi: `index_file(file_path, metadata, source_name=None)`.
> Doc_id `{stem}_{sha256[:16]}` formatında üretiliyor; **stem** orijinal
> filename'den hesaplanmalı (FastAPI tempfile path'inin rastgele
> stem'i `tmpXYZ` doc_id'yi farklılaştırırdı → duplicate-skip çalışmazdı).
> API tarafı `source_name=file.filename` geçiriyor; lokal script'te None
> default → file_path.name fallback.

> **📝 Implementation note (Sapma 33 — client DI):** `__init__`'e
> `client: QdrantClient | None = None` parametresi eklendi. Default None
> → URL'den yaratılır (geriye uyumlu); test'te `QdrantClient(":memory:")`
> geçirilir → ağ/disk yok.

> **📝 Implementation note (ek özellikler):** Doc-level dedup
> (`{stem}_{content_sha256[:16]}`), point ID = `uuid5(NAMESPACE, "{doc_id}:{chunk_index}")`
> — idempotent upsert. Payload metadata: `source`, `page_num`, `subject`,
> `chunk_index`, `doc_id`, `text` (chunk içeriği validator/UI için).
> `payload_index` `doc_id` üzerine — scroll filter O(log n).

### rag/retriever.py

```python
class EduRetriever:
    def retrieve(self, query: str, subject: str = None, k: int = 4) -> list[Document]:
        """
        - subject belirtilmişse sadece o dersten ara
        - k: kaç chunk döndür (default 4)
        - Her Document'ta: page_content + metadata (source, page, subject)
        """
    
    def get_context_string(self, docs: list[Document]) -> str:
        """Chunk'ları tek string'e birleştir (prompt için)"""
```

> **📝 Implementation note (Sapma 12-15):**
> - **Sync API** — SPEC örneklerde `await retriever.retrieve(...)` yazsa da
>   qdrant-client blocking → retrieve sync. LangGraph node async içinde
>   sync IO meşru.
> - **Document tipi** = `langchain_core.documents.Document` — LangGraph +
>   CrewAI ortak tüketici, ek wrapper sınıfı premature abstraction olur.
> - **Score** `metadata["score"]`'da saklanıyor (SPEC dönüş tipi
>   `list[Document]`'i bozmadan). Validator threshold için zorunlu.
> - **`get_context_string` formatı:** numaralı + Türkçe kaynak/sayfa header:
>   ```
>   [1] (kaynak: tarih_tanzimat.txt, sayfa: 0)
>   <chunk metni>
>   ```
>   Model alıntı yapması ("[1]'de söylendiği gibi...") + parse-friendly.

> **📝 Implementation note (Sapma 33):** `client` DI parametresi (indexer ile
> aynı pattern) — test'te in-memory.

### graph/state.py

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    question: str
    subject: str
    grade_level: int
    session_id: str
    
    retrieved_docs: list       # RAG sonuçları
    context: str               # retrieved_docs'tan oluşturulan bağlam
    
    answer: str                # üretilen cevap
    confidence: float          # 0.0-1.0
    sources: list[str]         # hangi dökümanlardan
    
    attempts: int              # kaç kez denendi
    needs_retry: bool          # tekrar denenecek mi?
    
    messages: Annotated[list, add_messages]  # konuşma geçmişi
```

> **📝 Implementation note (Sapma 16-17 — ek `graph/llm.py` modülü):**
> SPEC graph/ klasörü 4 dosya öneriyor (state, nodes, edges, pipeline);
> **5. dosya `llm.py` eklendi**. `LLMBackend` Protocol + 3 concrete sınıf
> (Anthropic / Qwen3Local / VLLM) + `get_llm()` factory. ENV `LLM_BACKEND`
> ile config-driven swap. Qwen3LocalBackend ve VLLMBackend `__init__`'te
> `NotImplementedError` (stub) — Task 5+'da Linux/CUDA path'inde implement.
> macOS dev'de Anthropic kullanılır.

### graph/nodes.py — 4 düğüm

```python
async def retrieve_node(state: AgentState) -> AgentState:
    """Soruya göre ilgili chunk'ları getir"""
    retriever = EduRetriever()
    docs = await retriever.retrieve(state["question"], state["subject"])
    return {**state, "retrieved_docs": docs, "context": retriever.get_context_string(docs)}

async def generate_node(state: AgentState) -> AgentState:
    """Fine-tuned model veya Claude ile cevap üret.

    Üç implementasyon seçeneği (P3 development'ta seçilecek):

    1) **P2 fine-tuned model (Qwen3-4B + LoRA):**
       - PeftModel.from_pretrained ile yükle (P3_HANDOFF.md § 3)
       - Context'i system message'a ekle: "Aşağıdaki bağlamdan yararlanarak cevap ver..."
       - tokenizer.apply_chat_template + model.generate(use_cache=True, do_sample=False)
       - **GPU şart** (T4'te 25-35 sn latency RAG context ile)

    2) **Anthropic API (geliştirme + production fallback):**
       - claude-haiku-4-5 hızlı + ucuz, dev döngüsünü uzatmaz
       - System: "Sen Türkçe lise eğitim asistanısın. Bağlamdan yararlanarak..."
       - User: question + retrieved context

    3) **vLLM serving (production):**
       - Adapter merge + 4-bit AWQ → vLLM endpoint
       - ~3-5 sn latency, batched serving
       - P3 sonu "production gateway" task'ında ele alınır
    """
    # Context + question → model → answer
    # Confidence'ı da hesapla (basit heuristic: cevap uzunluğu, belirsiz kelimeler vb.)
    return {**state, "answer": answer, "confidence": confidence, "attempts": state["attempts"] + 1}

async def validate_node(state: AgentState) -> AgentState:
    """Cevap kaliteli mi?"""
    # Basit kurallar:
    # - 50 karakterden kısa → yetersiz
    # - "bilmiyorum" içeriyor → retry
    # - 3. denemeyse zorla bitir
    needs_retry = len(state["answer"]) < 50 and state["attempts"] < 3
    return {**state, "needs_retry": needs_retry}

async def format_node(state: AgentState) -> AgentState:
    """Final cevabı formatla, source'ları ekle"""
    sources = [doc.metadata["source"] for doc in state["retrieved_docs"]]
    return {**state, "sources": list(set(sources))}
```

> **📝 Implementation note (Sapma 18-20, 30 — node davranışları):**
> - **Validator (Sapma 18):** length kontrolü + Türkçe belirsizlik kalıpları
>   (`"bilmiyorum"`, `"yeterli bilgi yok"`, `"emin değil"`, `"bağlamda yer
>   almıyor"`). MVP — Task 4/5 sonrası NLI/LLM-as-judge ile değiştirilebilir.
> - **Confidence (Sapma 20):** `top_retrieved_doc.metadata["score"]`. Cevap
>   uzunluğu yanıltıcı (uzun zayıf olabilir); retrieval skoru retrieval
>   kalitesini ölçer. Smoke test'te 0.89-0.91 değerleri.
> - **Sync retriever (Sapma 19):** SPEC örnekte `await retriever.retrieve(...)`
>   yazıyor ama retriever sync (Sapma 12 ile tutarlı); node async, içinde
>   sync IO çağrısı meşru.
> - **Sapma 21 ÇÖZÜLDÜ → Sapma 30:** İlk implementation'da `retrieve_node`
>   her çağrıda yeni `EduRetriever()` yaratıyordu → her request 30+ sn cold
>   start. Çözüm: `_retriever_singleton` module-level + `_get_retriever()`
>   helper. FastAPI lifespan'inde pre-warm yapılınca ilk request'te de
>   model warm. Test izolasyonu için autouse `reset_retriever_singleton`
>   fixture (Sapma 35).
>
> Ek olarak: her node'a `@_log_node(name)` decorator (structlog event'ler;
> giriş + çıkış elapsed_ms + updated_keys).

### graph/pipeline.py

```python
from langgraph.graph import StateGraph, END

def build_pipeline() -> StateGraph:
    graph = StateGraph(AgentState)
    
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("validate", validate_node)
    graph.add_node("format", format_node)
    
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "validate")
    graph.add_conditional_edges(
        "validate",
        lambda s: "generate" if s["needs_retry"] else "format"
    )
    graph.add_edge("format", END)
    
    return graph.compile()
```

### crew/agents.py — 2 agent

```python
"""
Researcher: RAG ile kaynak bulur
Writer: Öğrenci seviyesine uygun cevap yazar
"""

def create_researcher_agent(rag_tool) -> Agent:
    return Agent(
        role="Eğitim İçerik Araştırmacısı",
        goal="Soruyla ilgili en doğru ve güncel kaynakları bul",
        backstory="Türk lise müfredatında uzman...",
        tools=[rag_tool],
        verbose=True,
        max_iter=3,
    )

def create_writer_agent() -> Agent:
    return Agent(
        role="Pedagojik İçerik Yazarı",
        goal="Öğrenci seviyesine uygun, anlaşılır cevap yaz",
        backstory="10 yıllık lise öğretmeni deneyimi...",
        verbose=True,
    )
```

> **📝 Implementation note (Sapma 22 — F-2 ÇÖZÜLDÜ):** SPEC `crewai>=0.80.0`
> 2024 yazımı; 2026'da kurulan **CrewAI 1.14.3** (major bump 0→1). API
> uyumluluğu yüksek kaldı (kwargs Pydantic v2 driven), tek değişen
> **LLM provider config**: `from crewai import LLM` + LiteLLM wrapper.

> **📝 Implementation note (Sapma 23 — LiteLLM provider prefix):**
> `LLM(model="anthropic/claude-haiku-4-5")` — explicit provider prefix.
> LiteLLM auto-detect ambiguity önler. ANTHROPIC_API_KEY ENV otomatik
> çekilir.

> **📝 Implementation note (Sapma 24 — Writer hallucination ÇÖZÜLDÜ):**
> İlk smoke test'te Writer "Kaynaklar" bölümüne bağlamda olmayan kitap
> referansları (Newton 1687 *Principia Mathematica*, Osmanlı Arşivi)
> uydurmuştu. Fix prompt-level: writing task description'da explicit
> kurallar (kitap/yazar/tarih ekleme, sadece dosya adı), backstory'de
> "uydurma yayın bilgisi yazmazsın" eklendi. Re-test'te sıfır uydurma.
> Production'da post-validator (LangGraph validate pattern) eklenebilir.

> **📝 Implementation note (Sapma 25 — token cost):** CrewAI 3 LLM call
> (~8800 token / soru, ~$0.04 claude-haiku-4-5'te); LangGraph 1 call
> (~3000 token, ~$0.013). Maliyet 3x → CrewAI sadece **multi-disciplinary
> sorular** için route edilmeli; basit sorularda LangGraph yeter (Task 5
> endpoint routing planı).

### P1 API güncellemesi — yeni endpoint

```python
# services/api/app/routers/questions.py'e ekle

@router.post("/ask/v2")  # P3 versiyonu — gerçek AI
async def ask_question_v2(
    request: QuestionRequest,
    pipeline = Depends(get_agent_pipeline)
):
    """
    LangGraph pipeline üzerinden cevap üret.
    AgentState'i başlat, pipeline çalıştır, QuestionResponse döndür.
    """
```

> **📝 Implementation note (Sapma 26-31 — P1 entegrasyon detayları):**
>
> - **Sapma 26 — PYTHONPATH cascade:** Lokal dev başlangıcı:
>   ```bash
>   cd services/api
>   PYTHONPATH=$(pwd):$(pwd)/../.. uvicorn app.main:app --port 8000
>   ```
>   P1 `from app.X` (cwd) + agents `from agents.X` (repo root) iki yol.
>
> - **Sapma 27 — shared venv:** Lokal dev'de `.venv-agents`'a hem
>   `agents/requirements.txt` hem `services/api/requirements.txt` yüklenir
>   (production Docker'da services/api container'ı her ikisini handle eder).
>
> - **Sapma 28 — `chunks_indexed` field:** `DocumentUploadResponse.chunks_indexed:
>   int = 0` eklendi (TASKS.md ekstra gereksinimi). Default=0 geriye uyumlu;
>   duplicate-skip senaryosunda da 0.
>
> - **Sapma 29 — tempfile pattern:** `UploadFile` → `tempfile.NamedTemporaryFile`
>   → `indexer.index_file(tmp_path, source_name=file.filename)`. Indexer
>   FastAPI'den bağımsız kalır; tempfile try/finally'de silinir.
>
> - **Sapma 30 — lifespan eager init:** `main.py` lifespan startup'ta
>   `DocumentIndexer + build_pipeline + _get_retriever()` initialize edilir.
>   Embedder model (e5-large) startup'ta yüklenir → ilk request cold start
>   cezası yok (Sapma 21 ÇÖZÜLDÜ). Hata durumunda `app.state.indexer = None`
>   → endpoint'ler 503 verir, app yine ayakta (graceful degradation).
>
> - **Sapma 31 — `.env` cascade:** `_load_env_cascade()` Settings init'inden
>   ÖNCE çalışır; `agents/.env` → `<repo_root>/.env` → `ml/.env` sırasıyla
>   `load_dotenv` → ANTHROPIC_API_KEY tek dosyada tutmak yeter.
>
> - **`get_document_indexer` + `get_agent_pipeline`** dependencies — `Request`
>   injection ile `app.state`'ten döndürülür (`@lru_cache` ağır objelere
>   uygun değil); None ise 503 fırlatılır.

---

## docker-compose.yml güncellemesi

```yaml
services:
  api:
    ... (P1'den aynı)
  
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  qdrant_data:
```

> **📝 Implementation note (Sapma 1, 2, 5):**
> - **Qdrant `latest` → `v1.12.4` sabit pin (Sapma 1):** reproducibility;
>   `latest` minor schema değişikliklerinde compose'u kırar.
> - **Healthcheck eklendi (Sapma 5):** `wget --spider /readyz` + `start_period:
>   10s`. Task 5'te `api → depends_on: qdrant: condition: service_healthy`
>   eklenmesi için zemin (şu an depends_on yok — Sapma 2: standalone API
>   dev'i zorlamasın).
> - **api'a `QDRANT_URL=http://qdrant:6333` ENV** — Docker network DNS
>   üzerinden bağlanır (lokal dev'de `localhost:6333`).

> **📝 Implementation note (Sapma 9 — qdrant-client minor mismatch):**
> Floor pin `qdrant-client>=1.12` kurulduğunda 1.17.x çekiyor; server
> v1.12.4 ile minor diff ≥ 1 → ilk request uyarı atar
> (`Major versions should match...`). Pratikte uyumsuzluk gözlenmedi
> (smoke + tests passed). Fix path: server upgrade (`v1.13.x`) veya
> client pin (`<1.13`). Şimdilik uyarı sinyal olarak korunuyor.

---

## requirements.txt

```
# --- LangChain / LangGraph / CrewAI ---
langchain>=0.3.0
langchain-community>=0.3.0
langchain-text-splitters>=0.3.0
langgraph>=0.2.0
crewai>=0.80.0

# --- Vector DB + embeddings ---
qdrant-client>=1.12.0
sentence-transformers>=3.1.0
pypdf>=4.2.0

# --- LLM clients ---
# Anthropic — fallback / development LLM (Sapma 27'de Claude API zaten kullanıldı P2 için)
anthropic>=0.40.0

# --- P2 adapter inference (Qwen3-4B + LoRA) ---
# Eğer agent'lar P1 API container'ında çalışacaksa bu paketler gerekli.
# Geliştirme aşamasında Anthropic API ile başlanırsa skip edilebilir.
torch==2.9.1
transformers>=4.46
peft>=0.13
bitsandbytes>=0.44 ; sys_platform != 'darwin'
accelerate>=1.0

# --- Dev / CI ---
pytest>=8.3
ruff>=0.6
```

> **📝 Sürüm hassasiyeti uyarısı (P2'den taşınan ders):** Bu spec yazıldığında
> üretilen pin'ler 2024 sürümleriydi. **P2'de Sapma 5** ile relaxed pin'e (>=)
> geçildi çünkü Python 3.13'te eski torch wheel yok. P3'te de aynı yaklaşımı
> sürdür: floor pin (`>=`) + `requirements.lock.txt` (pip freeze) ile
> reproducibility. Ek olarak **Colab'da torch'u reinstall etme** (Sapma 26):
> native CUDA 12.8 build'i bozulur.

> **📝 Implementation note (Sapma 6, 8 — P3 requirements detayları):**
> - **+4 paket eklendi (Sapma 6):** `pytest-asyncio` (async node testleri),
>   `python-dotenv` (.env cascade), `pydantic-settings` (P1 config uyumu),
>   `structlog` (P1 logging stack uyumu).
> - **`torch==2.9.1` → `torch>=2.9` (Sapma 8):** sentence-transformers
>   2.11.0 çekti; macOS dev için sürüm uyumu kritik değil (bitsandbytes
>   `sys_platform != 'darwin'` guard). P2 ml/ pin'i Colab CUDA 12.x +
>   bitsandbytes ABI içindi; agents/ Anthropic API ile dev → torch sürümü
>   önemsiz. Production Linux + CUDA için `agents/requirements.lock.txt`
>   (pip freeze) eklendiğinde sabitlenir.

---

## Teslim kriterleri

- [x] `docker-compose up` → API + Qdrant birlikte çalışıyor
- [x] `POST /v1/documents/upload` → gerçekten Qdrant'a yüklüyor
- [x] `POST /v1/questions/ask/v2` → RAG cevabı + sources döndürüyor
- [x] LangGraph: validate başarısız olunca retry döngüsü çalışıyor
- [x] CrewAI: karmaşık sorular için 2 agent işbirliği yapıyor
- [x] Test: `pytest agents/tests/` geçiyor (11/11, 0.18 sn)
- [x] README: sistemi nasıl çalıştırırsın, döküman nasıl yüklersin

> **📝 Implementation note (Sapma 34-35 — test mimarisi):**
> Testler **in-memory Qdrant + FakeEmbedder + MockLLM** ile çalışır
> (gerçek e5-large 2 GB + Anthropic call CI'da uygunsuz):
> - **FakeEmbedder (Sapma 34a):** keyword-bazlı 16-dim L2-normalize vektör.
>   `TurkishEmbedder` protokolüne uyar (`embed_documents`, `embed_query`,
>   `vector_size`).
> - **MockLLM (Sapma 34b):** `LLMBackend` protocol; sıralı response stub
>   (retry + max_attempts test'leri için).
> - **`reset_retriever_singleton` autouse (Sapma 35):** `nodes.py`
>   module-level singleton'ı her test başında/sonunda None'a sıfırlar
>   → fixture izolasyonu.
> - **Indexer/Retriever `client` DI (Sapma 33):** Test'te
>   `QdrantClient(":memory:")` geçirilir; production None default.
>
> Sonuç: 11 test, 0.18 sn. Network/disk yok, deterministik.

> **📝 Implementation note (test_connection.py vector_size — Sapma 4):**
> Task 0 smoke test'inde `VECTOR_SIZE = 384` dummy. Embedding modeli
> Task 1'de seçileceği için (F-1 brainstorm) sembolik değer; gerçek
> collection (e5-large 1024-dim) Task 1'de yeniden oluşturuldu.
