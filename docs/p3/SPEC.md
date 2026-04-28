# P3 — Akıllı Ajan: Proje Spesifikasyonu

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

---

## Teslim kriterleri

- [ ] `docker-compose up` → API + Qdrant birlikte çalışıyor
- [ ] `POST /v1/documents/upload` → gerçekten Qdrant'a yüklüyor
- [ ] `POST /v1/questions/ask/v2` → RAG cevabı + sources döndürüyor
- [ ] LangGraph: validate başarısız olunca retry döngüsü çalışıyor
- [ ] CrewAI: karmaşık sorular için 2 agent işbirliği yapıyor
- [ ] Test: `pytest agents/tests/` geçiyor
- [ ] README: sistemi nasıl çalıştırırsın, döküman nasıl yüklersin
