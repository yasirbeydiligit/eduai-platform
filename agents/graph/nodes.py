"""LangGraph node'ları — retrieve / generate / validate / format.

Her node `AgentState` partial update'i döndürür (LangGraph reducer ile merge).
Node'lar `async def` — LangGraph fork/parallel için zorunlu, Anthropic API
beklemesi de async. Retriever sync (Sapma 12); node içinde direkt çağrılır.

Logging: her node giriş + çıkışında structlog event (TASKS.md gereksinimi,
P1 pattern). `_log_node` decorator tekrarlamayı azaltır.
"""

from __future__ import annotations

import functools
import time
from typing import Awaitable, Callable

import structlog

from agents.graph.llm import get_llm
from agents.graph.state import AgentState
from agents.rag.retriever import EduRetriever

logger = structlog.get_logger(__name__)

# --- Validator ayarları ---

# 3 denemeden sonra zorla bitir — sonsuz retry'ı engelle.
_MAX_ATTEMPTS = 3

# Cevap "yetersiz" sayan eşik (karakter).
_MIN_ANSWER_CHARS = 50

# Zayıf cevap göstergeleri — bunları içeren cevap retry tetikler.
# Türkçe pedagoji ton'una has belirsizlik kalıpları.
_WEAK_INDICATORS: tuple[str, ...] = (
    "bilmiyorum",
    "yeterli bilgi yok",
    "emin değil",
    "veriliş bilgilere göre değerlendirme yapamıyorum",
    "bağlamda yer almıyor",
)


def _is_weak_answer(answer: str) -> bool:
    """Cevap heuristic olarak zayıf mı? Length + weak indicator bayrağı.

    MVP — Task 4/5'te NLI veya LLM-as-judge ile değiştirilebilir
    (Sapma 18 plan).
    """
    if len(answer) < _MIN_ANSWER_CHARS:
        return True
    a = answer.lower()
    return any(ind in a for ind in _WEAK_INDICATORS)


# --- Logging decorator ---


def _log_node(name: str):
    """Node giriş + çıkış event'lerini structlog'a yazan decorator."""

    def wrap(
        fn: Callable[[AgentState], Awaitable[dict]],
    ) -> Callable[[AgentState], Awaitable[dict]]:
        @functools.wraps(fn)
        async def inner(state: AgentState) -> dict:
            t0 = time.perf_counter()
            logger.info(
                "node_enter",
                node=name,
                session=state.get("session_id"),
                attempts=state.get("attempts", 0),
            )
            result = await fn(state)
            elapsed = time.perf_counter() - t0
            # Çıkışta hangi key'lerin update edildiğini logla — debug ergonomisi.
            logger.info(
                "node_exit",
                node=name,
                seconds=round(elapsed, 3),
                updated_keys=sorted(result.keys()),
            )
            return result

        return inner

    return wrap


# --- Node'lar ---


# Module-level retriever singleton — Sapma 21 fix.
# İlk `retrieve_node` çağrısında oluşturulur, sonraki tüm çağrılarda paylaşılır.
# Lifespan'de `_get_retriever()` çağrısı yapılırsa (FastAPI Task 5) ilk
# request'in cold-start cezasını da elimine eder.
# Test reset: `agents.graph.nodes._retriever_singleton = None`.
_retriever_singleton: EduRetriever | None = None


def _get_retriever() -> EduRetriever:
    """Module-level retriever — lifespan/startup'ta pre-warm için public-ish."""
    global _retriever_singleton
    if _retriever_singleton is None:
        _retriever_singleton = EduRetriever()
    return _retriever_singleton


@_log_node("retrieve")
async def retrieve_node(state: AgentState) -> dict:
    """Kullanıcı sorusunu Qdrant retriever'la 4 chunk'a indir.

    Retriever sync (Sapma 12) → direkt çağrılır; LangGraph node async olsa da
    içinde sync IO kabul. Singleton retriever Sapma 21 fix.
    """
    retriever = _get_retriever()
    docs = retriever.retrieve(
        query=state["question"],
        subject=state.get("subject"),
        k=4,
    )
    context = retriever.get_context_string(docs)
    return {"retrieved_docs": docs, "context": context}


@_log_node("generate")
async def generate_node(state: AgentState) -> dict:
    """Bağlam + soruyu LLM'e ver, cevap + güven skoru üret.

    Confidence MVP heuristic'i: top retrieved doc'un score'u (Sapma 20).
    """
    llm = get_llm()
    answer = await llm.generate(
        question=state["question"],
        context=state.get("context", ""),
        max_tokens=512,
    )

    # Güven skoru = en yakın chunk'ın cosine similarity'si.
    # Validator threshold için Task 3 MVP; ileride LLM-as-judge ile değişebilir.
    docs = state.get("retrieved_docs", [])
    confidence = float(docs[0].metadata.get("score", 0.0)) if docs else 0.0

    return {
        "answer": answer,
        "confidence": confidence,
        "attempts": state.get("attempts", 0) + 1,
    }


@_log_node("validate")
async def validate_node(state: AgentState) -> dict:
    """Cevabı kalite kurallarına karşı denetle, retry kararı ver.

    Kurallar (MVP):
      - <50 karakter → zayıf
      - "bilmiyorum" / "yeterli bilgi yok" gibi belirsizlik → zayıf
      - 3 deneme sonrası zorla bitir (sonsuz retry yok)
    """
    answer = state.get("answer", "")
    attempts = state.get("attempts", 0)
    needs_retry = attempts < _MAX_ATTEMPTS and _is_weak_answer(answer)
    return {"needs_retry": needs_retry}


@_log_node("format")
async def format_node(state: AgentState) -> dict:
    """Cevap onaylandı, dönüş için kaynak listesini topla."""
    docs = state.get("retrieved_docs", [])
    # Set ile unique → sorted ile deterministik dönüş (UI tutarlılığı).
    sources = sorted(
        {doc.metadata.get("source", "?") for doc in docs if doc.metadata.get("source")}
    )
    return {"sources": sources}
