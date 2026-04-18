# EduAI API

[![CI](https://github.com/yasirbeydiligit/eduai-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/yasirbeydiligit/eduai-platform/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

A production-ready FastAPI backend for a Turkish-language AI education platform targeting high-school curricula.
This is **P1** of a four-phase learning project that builds an end-to-end AI-powered tutoring system — from HTTP scaffolding to fine-tuned models, retrieval-augmented agents, and cloud deployment.
The current phase delivers the architectural foundation (routing, validation, containerization, CI); AI integration arrives in P3.

## Tech Stack

| Technology | Why it's here |
|------------|---------------|
| **Python 3.11+** | Modern language features (`StrEnum`, `datetime.UTC`, `match`/`case`) and performance improvements |
| **FastAPI 0.111+** | Async-first, Pydantic-integrated OpenAPI generation, fastest-in-class for Python web APIs |
| **Pydantic v2** | Rust-backed validation (~10× faster than v1), rich field constraints, `Annotated` pattern support |
| **Uvicorn[standard]** | Production ASGI server with `uvloop` + `httptools` for throughput |
| **structlog** | Structured JSON logging — queryable fields (`question_id`, `session_id`) in production aggregators |
| **pytest + httpx** | FastAPI's official `TestClient` stack; async-aware, no real HTTP server needed |
| **ruff** | Single Rust-based tool replacing `flake8`/`isort`/`black` — ~100× faster, richer rule set |
| **Docker (multi-stage)** | Slim production image (~284 MB), non-root user, builder/runtime separation |
| **GitHub Actions** | Three-stage pipeline: lint → test → docker-build; fail-fast with `needs:` dependencies |

## Quick Start

**Prerequisites:** Docker 20.10+, Python 3.11+ (only needed for running tests outside Docker)

```bash
git clone https://github.com/yasirbeydiligit/eduai-platform.git
cd eduai-platform
docker-compose up --build
```

Once the container is up, open **http://localhost:8000/docs** for the interactive Swagger UI.

Run tests locally (optional):

```bash
cd services/api
pip install -r requirements-dev.txt
pytest tests/ -v
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service liveness + version + dependency status |
| `POST` | `/v1/sessions/` | Create a new session (returns UUID) |
| `GET` | `/v1/sessions/{session_id}` | Retrieve session summary (question count, subjects accessed) |
| `POST` | `/v1/questions/ask` | Submit a question; returns a mock answer in P1 |
| `POST` | `/v1/documents/upload` | Upload a PDF or TXT reference document (max 10 MB) |

Full OpenAPI schema: `http://localhost:8000/openapi.json`

## Example Requests

### Health check

```bash
curl http://localhost:8000/health
```

### Create a session

```bash
curl -X POST http://localhost:8000/v1/sessions/
# → {"session_id": "550e8400-...", "created_at": "2026-04-18T10:00:00+00:00"}
```

### Ask a question

```bash
curl -X POST http://localhost:8000/v1/questions/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Osmanlı İmparatorluğu ne zaman kuruldu?",
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "subject": "tarih",
    "grade_level": 9
  }'
```

### Retrieve session details

```bash
curl http://localhost:8000/v1/sessions/550e8400-e29b-41d4-a716-446655440000
```

### Upload a document

```bash
curl -X POST http://localhost:8000/v1/documents/upload \
  -F "file=@lecture_notes.pdf" \
  -F "title=Ottoman History Chapter 1" \
  -F "subject=tarih" \
  -F "grade_level=9"
```

## Project Structure

```
eduai-platform/
├── services/
│   └── api/                           # P1 — FastAPI backend
│       ├── app/
│       │   ├── main.py                # App entry, lifespan, CORS, /health
│       │   ├── core/config.py         # Pydantic Settings (env-driven)
│       │   ├── routers/               # HTTP layer — thin, delegates to services
│       │   ├── schemas/               # Pydantic request/response models
│       │   ├── services/              # Business logic — framework-agnostic
│       │   └── dependencies.py        # FastAPI DI factories (singleton services)
│       ├── tests/                     # pytest + TestClient (13 tests)
│       ├── Dockerfile                 # Multi-stage, non-root user, healthcheck
│       ├── requirements.txt           # Production dependencies
│       └── requirements-dev.txt       # Adds pytest, ruff, httpx
├── docs/
│   └── p1/
│       ├── CONCEPT.md                 # Learning notes — FastAPI, Docker, CI concepts
│       ├── SPEC.md                    # Implementation specification
│       ├── TASKS.md                   # Session-sized task breakdown
│       └── IMPLEMENTATION_NOTES.md    # Intentional deviations from SPEC, with rationale
├── .github/workflows/ci.yml           # Lint → test → docker-build pipeline
├── docker-compose.yml                 # Dev: hot reload, healthcheck, bind-mount
└── README.md
```

**Architectural discipline:**
- Routers handle HTTP only — parsing, status codes, exceptions. No business logic.
- Services are HTTP-agnostic; they can be reused from CLI tools, background jobs, or other transports.
- Schemas live in their own layer — single source of truth for validation and OpenAPI docs.
- `docs/p1/IMPLEMENTATION_NOTES.md` captures every deliberate deviation from the spec with the reasoning — useful context for anyone (or any AI assistant) picking up the codebase.

## Project Context

This repository is phase one of a structured four-phase curriculum for building a production AI platform end-to-end:

| Phase | Focus | Key technologies |
|-------|-------|------------------|
| **P1** (current) | HTTP API scaffold, mock answers, CI/CD | FastAPI · Docker · GitHub Actions |
| **P2** | Fine-tuning pipeline for Turkish education content | PyTorch · LoRA/QLoRA · MLflow |
| **P3** | Retrieval-augmented agents | LangChain · Qdrant (RAG) · LangGraph · CrewAI |
| **P4** | Production deployment | Kubernetes · AWS EKS · Terraform · observability |

The mock answers returned by `/v1/questions/ask` in P1 will be replaced with real model inference in P3 — the service interface is already shaped to support that transition without touching router or test code.

## Learning Goals

This project is an intentional vehicle for practicing senior-level backend engineering skills:

- Designing FastAPI applications with strict layer separation (routing vs. business logic vs. schema)
- Writing Pydantic v2 schemas with custom validators, `Annotated` dependency injection, and `StrEnum`
- Building multi-stage Docker images with non-root users and healthchecks
- Splitting runtime and development dependencies for slim production images
- Writing GitHub Actions pipelines with job dependencies and pip caching
- Test-driving API endpoints with `TestClient`, including multipart file upload paths
- Documenting architectural trade-offs as they happen (`IMPLEMENTATION_NOTES.md`) rather than retrofitting them

## License

MIT (or the owner's choice — no license file committed yet; treat as "all rights reserved" until one is added).
