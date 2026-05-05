# P5 — Backend Production: Spesifikasyon

---

## Klasör yapısı (eklemeler)

```
services/
├── api/app/
│   ├── auth/                    ← YENİ: JWT + bcrypt
│   │   ├── jwt_handler.py
│   │   ├── password.py
│   │   └── dependencies.py      ← get_current_user
│   ├── db/                      ← YENİ: SQLAlchemy + Alembic
│   │   ├── base.py
│   │   ├── models/{user,session,message,usage_log}.py
│   │   └── session.py           ← async engine
│   ├── routers/{auth,users}.py  ← YENİ
│   └── services/
│       ├── rate_limiter.py      ← slowapi + per-user cost
│       └── llm_router.py        ← Hybrid Anthropic ↔ vLLM
├── vllm/                        ← YENİ: vLLM serving container
│   ├── Dockerfile
│   ├── start.sh
│   └── lora_adapters/           ← P2 LoRA kopya
└── postgres/alembic/            ← migrations
```

---

## DB models

```python
class User(Base):
    id, email (unique), password_hash, created_at, plan_tier, kvkk_consent_version

class Session(Base):
    id, user_id (FK), subject, grade_level, started_at

class Message(Base):
    id, session_id (FK), role (user|assistant), content, tokens_used, created_at

class UsageLog(Base):
    id, user_id, model_name, input_tokens, output_tokens, cost_cents, created_at
```

---

## Endpoint'ler

| Endpoint | Method | Açıklama |
|---|---|---|
| `/v1/auth/signup` | POST | email + password + kvkk_consent → User + JWT |
| `/v1/auth/login` | POST | email + password → JWT |
| `/v1/auth/refresh` | POST | refresh token → access token |
| `/v1/users/me` | GET | current user info |
| `/v1/users/me` | DELETE | cascade + 30d buffer |
| `/v1/users/me/export` | GET | JSON tüm data |
| `/v1/sessions/{id}/messages` | GET | history fetch |
| `/v1/questions/ask/v2` | POST | (mevcut) + persist + cost log |
| `/metrics` | GET | Prometheus |

---

## requirements.txt eklemeleri

```
sqlalchemy[asyncio]>=2.0
asyncpg>=0.29
alembic>=1.13
python-jose[cryptography]>=3.3
passlib[bcrypt]>=1.7.4
slowapi>=0.1.9
prometheus-fastapi-instrumentator>=7.0
```

---

## docker-compose ekleme

```yaml
services:
  postgres:
    image: postgres:17-alpine
    environment:
      - POSTGRES_DB=eduai
      - POSTGRES_USER=eduai
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data

  vllm:
    build: ./services/vllm
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    profiles: ["production"]

volumes:
  postgres_data:
```

---

## vLLM Dockerfile (özet)

```dockerfile
FROM vllm/vllm-openai:latest
COPY lora_adapters/ /lora/
ENTRYPOINT ["vllm", "serve", "Qwen/Qwen3-4B-Instruct-2507",
    "--enable-lora", "--lora-modules", "eduai=/lora/eduai_qwen3"]
```

---

## Teslim kriterleri

- [ ] Postgres migration tek komutla (`alembic upgrade head`)
- [ ] Signup → login → /ask/v2 zinciri JWT ile
- [ ] Rate limit aşan user 429
- [ ] User delete → cascade message/session
- [ ] vLLM container hybrid router ile çalışıyor
- [ ] Prometheus `/metrics` açık
- [ ] `docs/p5/KVKK.md` politika hazır
- [ ] pytest 50+ PASSED
- [ ] `docs/p5/P6_HANDOFF.md` (Frontend kickoff paketi)
