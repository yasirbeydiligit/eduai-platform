# P5 — Backend Production: Kavramsal Kılavuz

> P4'te içerik + math agent kuruldu. P5 **production altyapısı**: Postgres
> conversation persistence, Auth/multi-tenant, Rate limit + cost control,
> KVKK uyumu, vLLM serving (kendi Qwen3 self-hosted).

---

## P5'in varlık nedeni

P3-P4 sonu çekirdek değer kanıtlandı (eval Δ pozitif). Ama:
- ❌ Conversation history her uvicorn restart'ında siliniyor (in-memory)
- ❌ Anyone API'yi çağırıyor (auth yok)
- ❌ Rate limit yok → tek user API kotasını yiyebilir
- ❌ KVKK uyumu yok (öğrenci verisi koruması Türkiye'de zorunlu)
- ❌ LLM cost user'a atfedilemiyor (per-user tracking)
- ❌ Anthropic API'ye sürekli para ödüyoruz, kendi model devre dışı

P5 bu 6 sınırı çözer.

---

## Büyük resim

```
P3 — In-memory dev API           P5 — Production-ready
────────────────────             ─────────────────────
QuestionService dict             ✅ Postgres + SQLAlchemy 2.x async
session_id mock                  ✅ JWT auth + bcrypt
ANTHROPIC_API_KEY .env           ✅ Multi-tenant + per-user rate limit
Anthropic-only LLM               ✅ Hybrid LLM router (Anthropic + vLLM Qwen3)
                                 ✅ KVKK: data retention, user delete, audit
                                 ✅ Observability (Prometheus + Grafana)
```

---

## Temel kavramlar (özet)

### 1. Postgres + Conversation Persistence
- `users` (auth), `sessions` (chat sessions), `messages` (history), `usage_log` (cost)
- SQLAlchemy 2.x async + Alembic migrations
- LangGraph state.messages persist edilir

### 2. JWT Auth — self-hosted
- `python-jose` + `passlib[bcrypt]`
- Access token (1h) + refresh token (30d)
- `get_current_user` dependency tüm `/v1/*` route'larda

### 3. Rate Limit + Cost Control
- `slowapi` per-IP (DDoS)
- Per-user günlük cost eşiği (free tier: 50 q/gün, $0.10/gün)
- `usage_log` üzerinden model_pricing × token = cents

### 4. KVKK — 6698 Sayılı Kanun
- Açık rıza signup'ta (versionlı)
- User delete cascade + 30d buffer
- Audit log (kim ne zaman ne gördü)
- Data export JSON
- Yurtdışı aktarım uyarısı (Anthropic US sunucusu)

### 5. vLLM Serving (Self-Hosted Qwen3)
- P2 LoRA adapter Drive'dan kopya
- AWS g4dn.xlarge spot ~$115/ay (production)
- LangGraph VLLMBackend stub'ı (P3 Sapma 17) implement
- **Avantaj:** veri dışarı çıkmaz (KVKK ✓) + sınırsız request

### 5b. Ollama lokal dev backend (macOS Metal native)

**Sorun:** vLLM Linux/CUDA-only → macOS dev makinesinde Anthropic API zorunlu
oluyor. Kullanıcı hedefi: dev iterasyonda da kendi model.

**Çözüm:** Ollama lokal serving:
- macOS Metal native (M-series GPU hızlı, ~3-5 sn/cevap)
- Bedava (sabit GPU yok, lokal makine)
- KVKK ✓ (veri dışarı çıkmaz)
- OpenAI compat API → mevcut `llm.py` AnthropicBackend benzeri OllamaBackend
- LoRA adapter merge + GGUF conversion gerek:
  ```bash
  # P2 LoRA → base model merge (PeftModel.merge_and_unload)
  # → llama.cpp convert.py → Q5_K_M GGUF
  # → ollama create eduai-qwen3 -f Modelfile
  ```

**Yeni `LLM_BACKEND` değeri:** `ollama-local` (anthropic / qwen3-vllm / ollama-local).

**Kullanım önceliği:**
- Test'lerde: **MockLLM** (deterministik, hızlı, CI uyumlu — değişmez)
- Dev iterasyon: opsiyonel Ollama (yavaş ama bedava + KVKK)
- Eval: cloud vLLM (production gerçeği)
- Production: cloud vLLM (sabit GPU + multi-user)

### 6. Hybrid LLM Router
- Heuristic complexity classifier (LLM gerekmez)
- basit (<50 char, selamlama) → Anthropic
- karmaşık (>200 char, math/code) → vLLM Qwen3
- Cost optimization: ölçek arttıkça hibrit kazanır

### 7. Observability
- Prometheus FastAPI middleware
- Custom: `llm_call_total{model}`, `llm_cost_cents_total{user_id}`
- Grafana dashboard JSON

---

## Cost gerçekliği

| Senaryo | Aylık |
|---|---|
| **Anthropic-only** (10k req/ay, claude-haiku) | ~$50 |
| **Kendi GPU only** (g4dn.xlarge spot 24/7) | ~$115 |
| **Hybrid** (50% basit Anthropic + 50% ağır Qwen3) | ~$155 |

İlk 1000 user altında **Anthropic ucuz**. Scale arttıkça **kendi GPU kazanır**.
KVKK için kendi GPU otomatik avantaj.

---

## P3-P4'ten taşınan dersler

1. **Lifespan eager init** — Postgres pool, vLLM client startup'ta hazır
2. **ENV cascade** — agents/.env + production secret manager (AWS SSM)
3. **DI pattern** — get_db_session, get_current_user
4. **Test in-memory** — pytest'te SQLite memory + mock JWT
5. **Sapma defteri** — docs/p5/IMPLEMENTATION_NOTES.md devam
