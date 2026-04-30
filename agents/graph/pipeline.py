"""LangGraph pipeline — retrieve → generate → validate → (retry|format) → END.

`build_pipeline()` compiled `StateGraph` döndürür; FastAPI dependency
(Task 5) bu fonksiyonu çağırıp app state'e cache'ler.

main test (`python -m agents.graph.pipeline`): TASKS.md gereksinimi.
ENV cascade'i kullanılarak ANTHROPIC_API_KEY otomatik yüklenir
(agents/.env → repo_root/.env → ml/.env). dotenv ile.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import structlog
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from agents.graph.edges import needs_retry_router
from agents.graph.nodes import (
    format_node,
    generate_node,
    retrieve_node,
    validate_node,
)
from agents.graph.state import AgentState

logger = structlog.get_logger(__name__)


def _load_env_cascade() -> None:
    """Birden fazla .env yolu dener; ilk var olanı yükler.

    Sıra:
      1. agents/.env       (P3 modülüne özel — tercih edilen)
      2. <repo_root>/.env  (proje-wide)
      3. ml/.env           (P2'den geriye uyum; ANTHROPIC_API_KEY zaten orada)

    Tüm dosyalar denenir; daha sonra yüklenenler önceki değerleri override
    etmez (`load_dotenv` default davranışı). Yani sıra önemli.
    """
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "agents" / ".env",
        repo_root / ".env",
        repo_root / "ml" / ".env",
    ]
    for path in candidates:
        if path.exists():
            load_dotenv(path)
            logger.debug("env_loaded", source=str(path))


def build_pipeline():
    """LangGraph state machine'i kur ve compile et.

    Akış:
        START → retrieve → generate → validate
                                        ↓
                          (needs_retry) Y → generate (retry)
                                        N → format → END

    Returns:
        Compiled `StateGraph` — `await pipeline.ainvoke(state)` ile çağrılır.
    """
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("validate", validate_node)
    graph.add_node("format", format_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "validate")
    # Conditional: validate sonrası ya retry ya format.
    # Mapping kontratı: router'ın döndürdüğü string → node adı.
    graph.add_conditional_edges(
        "validate",
        needs_retry_router,
        {"generate": "generate", "format": "format"},
    )
    graph.add_edge("format", END)

    return graph.compile()


# --- main test (TASKS.md gereksinimi) ---


async def _smoke() -> int:
    """Tek soru smoke test'i — TASKS.md test state'iyle.

    Beklenen: pipeline çalışır, answer + sources dolu döner. Anthropic
    backend için ANTHROPIC_API_KEY .env üzerinden yüklü olmalı.
    """
    _load_env_cascade()

    if (
        not os.getenv("ANTHROPIC_API_KEY")
        and os.getenv("LLM_BACKEND", "anthropic") == "anthropic"
    ):
        print(
            "✗ ANTHROPIC_API_KEY bulunamadı. Şu yollardan birine yaz:\n"
            "    agents/.env  |  ./.env  |  ml/.env\n"
            "Veya: export ANTHROPIC_API_KEY=sk-ant-..."
        )
        return 1

    pipeline = build_pipeline()

    state: AgentState = {
        "question": "Tanzimat Fermanı nedir?",
        "subject": "tarih",
        "grade_level": 9,
        "session_id": "test-123",
        "attempts": 0,
        "needs_retry": False,
    }

    print(f"Soru: {state['question']!r}")
    print(f"Subject: {state['subject']}, Sınıf: {state['grade_level']}")
    print(f"LLM Backend: {os.getenv('LLM_BACKEND', 'anthropic')}")
    print()

    result = await pipeline.ainvoke(state)

    print("=" * 80)
    print("ANSWER:")
    print(result.get("answer", "(boş)"))
    print()
    print("=" * 80)
    print(f"SOURCES: {result.get('sources', [])}")
    print(f"CONFIDENCE: {result.get('confidence', 0.0):.4f}")
    print(f"ATTEMPTS: {result.get('attempts', 0)}")
    print(f"NEEDS_RETRY (final): {result.get('needs_retry', False)}")

    if not result.get("answer"):
        print("\n  ✗ Cevap boş — pipeline akışı kontrol et.")
        return 1
    if not result.get("sources"):
        print("\n  ⚠ Sources boş — retriever 0 sonuç vermiş olabilir.")

    print("\n✓ Pipeline smoke test PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_smoke()))
