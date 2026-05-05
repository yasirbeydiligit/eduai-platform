# P5 — Görev Listesi: Backend Production

> P4 FINALIZED. Transfer paketi: `docs/p4/P5_HANDOFF.md`.

---

## Task 0 — Postgres + Alembic ⏱ ~45 dk

```
1. docker-compose: postgres servisi (postgres:17-alpine)
2. services/api/app/db/{base.py, session.py} — async SQLAlchemy 2.x
3. alembic init + boş migration
4. ENV: POSTGRES_PASSWORD, DATABASE_URL
5. Smoke: alembic upgrade head + connection test
```

## Task 1 — User model + JWT auth ⏱ ~1.5 saat

```
1. db/models/user.py
2. auth/{password.py, jwt_handler.py, dependencies.py}
3. routers/auth.py: signup/login/refresh
4. Migration: users table
5. Smoke: curl signup → login → token verify
```

## Task 2 — Conversation persistence ⏱ ~1 saat

```
1. db/models/{session.py, message.py}
2. routers/questions.py: /ask/v2 → message persist
3. P3 LangGraph state.messages → Postgres
4. GET /v1/sessions/{id}/messages
5. Smoke: 3 mesajlık konuşma → restart → history kalıcı
```

## Task 3 — Rate limit + cost tracking ⏱ ~1 saat

```
1. services/rate_limiter.py: slowapi per-IP
2. db/models/usage_log.py + middleware
3. cost_calculator: model_pricing dict
4. Per-user günlük eşik
5. Smoke: free tier 51. soruda 429
```

## Task 4 — KVKK ⏱ ~1 saat

```
1. signup'ta kvkk_consent_version
2. DELETE /v1/users/me cascade + 30d buffer
3. GET /v1/users/me/export JSON
4. docs/p5/KVKK.md politika
5. Audit log middleware
```

## Task 5 — vLLM serving ⏱ ~2 saat

```
1. services/vllm/Dockerfile + start.sh
2. P2 LoRA adapter kopya
3. docker-compose vllm (profiles: production)
4. agents/graph/llm.py: VLLMBackend implement (P3 Sapma 17 fix)
5. Smoke (Linux/CUDA dev veya AWS g4dn.xlarge):
   docker compose up vllm
   curl /v1/chat/completions test
```

## Task 5b — Ollama lokal dev backend (macOS) ⏱ ~1.5 saat

**Amaç:** Dev iterasyonda Anthropic API yerine kendi modeli kullan
(KVKK ✓ + bedava). Test'lerde MockLLM korunur.

```
1. P2 LoRA adapter (Drive'dan) → base model merge
   (peft.PeftModel.merge_and_unload + safetensors save)
2. llama.cpp convert.py: HuggingFace → GGUF Q5_K_M (kalite/boyut dengesi)
3. Ollama Modelfile yaz + ollama create eduai-qwen3 -f Modelfile
4. agents/graph/llm.py: OllamaBackend class (AnthropicBackend benzeri,
   OpenAI compat: http://localhost:11434/v1/)
5. ENV LLM_BACKEND=ollama-local desteği
6. Smoke: ollama run eduai-qwen3 → "Tanzimat ne zaman ilan edildi?" 
   → cevap ~3-5 sn macOS M-series'da
```

## Task 6 — Hybrid LLM router ⏱ ~45 dk

```
1. services/llm_router.py: HybridLLMRouter
2. Heuristic classifier (basit/karmaşık)
3. ENV LLM_ROUTING_STRATEGY=hybrid|anthropic|qwen3-local
4. Smoke: 5 soru routing log
```

## Task 7 — Observability ⏱ ~45 dk

```
1. prometheus-fastapi-instrumentator
2. /metrics endpoint
3. Custom metrics: llm_call_total, llm_cost_cents_total
4. Grafana dashboard JSON: infra/grafana/dashboards/api.json
5. docker-compose: prometheus + grafana (profiles: monitoring)
```

## Task 8 — Test suite + finalize ⏱ ~1.5 saat

```
1. test_auth.py
2. test_persistence.py
3. test_rate_limit.py
4. test_kvkk.py
5. test_llm_router.py
pytest 50+ PASSED
```

---

## P5 tamamlandı mı?

```
[ ] Postgres + auth + persistence çalışıyor
[ ] Rate limit + cost tracking aktif
[ ] KVKK delete + export açık
[ ] vLLM hybrid router (Linux/CUDA)
[ ] Prometheus + Grafana
[ ] pytest 50+ PASSED
[ ] docs/p5/KVKK.md + IMPLEMENTATION_NOTES_ARCHIVED.md
[ ] docs/p5/P6_HANDOFF.md
```
