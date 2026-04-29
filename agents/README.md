# EduAI Agents (P3)

RAG + LangGraph state machine + CrewAI multi-agent orchestration. P2 fine-tuned
Qwen3-4B adapter'ını RAG context'iyle birleştiren akıllı ajan katmanı.

> Spec: [`docs/p3/SPEC.md`](../docs/p3/SPEC.md) ·
> Tasks: [`docs/p3/TASKS.md`](../docs/p3/TASKS.md) ·
> Sapma defteri: [`docs/p3/IMPLEMENTATION_NOTES.md`](../docs/p3/IMPLEMENTATION_NOTES.md)

## Klasör yapısı

```
agents/
├── rag/                    # Embedding modeli, indexer, retriever (Task 1-2)
├── graph/                  # LangGraph state, nodes, edges, pipeline (Task 3)
├── crew/                   # CrewAI agent + task tanımları (Task 4)
├── memory/                 # Konuşma hafızası (session-bazlı)
├── tests/                  # pytest test'leri (Task 6)
├── test_connection.py      # Qdrant smoke test (Task 0 doğrulama)
├── requirements.txt        # Floor-pin bağımlılıklar (P2 pattern'i)
└── README.md
```

## Kurulum

```bash
# 1) Sanal ortam (lokal'de). P1/P2 venv'leri ile karışmaması için ayrı tut.
python -m venv .venv-agents
source .venv-agents/bin/activate

# 2) Tüm bağımlılıklar (CPU/macOS için; bitsandbytes Linux/CUDA only).
pip install -r agents/requirements.txt

# Alternatif (sadece Task 0 smoke test için minimal):
# pip install qdrant-client
```

## Task 0 — Qdrant smoke test

```bash
# 1) Qdrant container'ı ayağa kaldır
docker-compose up qdrant -d

# 2) Smoke test
python agents/test_connection.py

# Beklenen: tüm adımlar ✓ + "Smoke test PASSED"
# Qdrant Web UI: http://localhost:6333/dashboard
```

Smoke test başarılıysa Task 1'e (RAG indexer) geçilir.

## Geliştirme döngüsü notları (P2'den taşınan dersler)

- **LLM seçimi config-driven (`LLM_BACKEND`):** dev'de `anthropic` (claude-haiku-4-5,
  hızlı), prod'da `qwen3-local` veya `vllm`. P2 lokal inference T4'te 25+ sn —
  ajan döngüsünü iterasyonda yavaşlatır.
- **Smoke test maliyeti düşük, değeri yüksek:** her yeni component (Qdrant,
  LangGraph, CrewAI) önce minimal smoke. Full integration sonra.
- **Sapma yaz:** spec'le ayrılan her karar `docs/p3/IMPLEMENTATION_NOTES.md`'ye
  gerekçesiyle.
- **CI gate disiplini:** commit öncesi `ruff check + ruff format --check + pytest`.
