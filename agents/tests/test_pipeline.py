"""LangGraph pipeline akış testleri — full path + retry + max attempts.

Test stratejisi:
- Retriever singleton'ı populated retriever ile override (conftest reset).
- LLM factory `monkeypatch.setattr` ile MockLLM döndürmeye zorla.
- Pipeline `ainvoke()` ile çalıştır, state delta'larını assert et.

`pytest-asyncio` kullanıyoruz (agents/requirements.txt yüklü). Her async
test'e explicit `@pytest.mark.asyncio` markı.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.graph.pipeline import build_pipeline
from agents.graph.state import AgentState
from agents.rag.indexer import DocumentIndexer
from agents.rag.retriever import EduRetriever
from agents.tests.conftest import MockLLM


@pytest.fixture
def populated_retriever(
    indexer: DocumentIndexer,
    retriever: EduRetriever,
    tmp_path: Path,
) -> EduRetriever:
    """Tarih korpusu yüklenmiş retriever — pipeline test fixture'ı."""
    file_path = tmp_path / "tarih.txt"
    file_path.write_text(
        "Tanzimat Fermanı 1839'da Sultan Abdülmecid döneminde ilan edildi. "
        "Ferman modernleşme sürecinin başlangıcıdır. "
        "Mustafa Reşit Paşa tarafından okundu.",
        encoding="utf-8",
    )
    indexer.index_file(file_path, metadata={"subject": "tarih"})
    return retriever


def _install_pipeline_mocks(
    retriever: EduRetriever,
    mock_llm: MockLLM,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retriever singleton + LLM factory patch'i atomic apply."""
    import agents.graph.nodes as nodes_module

    nodes_module._retriever_singleton = retriever
    # nodes.py içinde `from agents.graph.llm import get_llm` import → bu
    # sembol nodes namespace'inde de görünür. monkeypatch nodes.get_llm doğru.
    monkeypatch.setattr(nodes_module, "get_llm", lambda: mock_llm)


def _initial_state(question: str = "Tanzimat Fermanı nedir?") -> AgentState:
    """TASKS.md test state'i — minimal initial fields."""
    return {
        "question": question,
        "subject": "tarih",
        "grade_level": 9,
        "session_id": "test-session-id",
        "attempts": 0,
        "needs_retry": False,
    }


@pytest.mark.asyncio
async def test_full_pipeline(
    populated_retriever: EduRetriever,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Soru → cevap + sources döner (tek seferde geçer, retry yok)."""
    mock_llm = MockLLM(
        [
            "Tanzimat Fermanı 1839'da ilan edildi. Bu cevap yeterince uzun ve detaylı.",
        ]
    )
    _install_pipeline_mocks(populated_retriever, mock_llm, monkeypatch)

    pipeline = build_pipeline()
    result = await pipeline.ainvoke(_initial_state())

    assert result["answer"], "Cevap boş olmamalı"
    assert "Tanzimat" in result["answer"]
    assert result["sources"], "Sources boş olmamalı"
    assert "tarih.txt" in result["sources"]
    assert result["attempts"] == 1, "Tek seferde geçmeliydi"
    assert result["needs_retry"] is False
    assert result["confidence"] > 0.0
    # MockLLM tek call almış olmalı.
    assert mock_llm.call_count == 1


@pytest.mark.asyncio
async def test_retry_logic(
    populated_retriever: EduRetriever,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """İlk cevap kısa → retry → ikinci cevap uzun → format'a geçer."""
    mock_llm = MockLLM(
        [
            "kısa",  # <50 char → validator weak → retry
            "Yeterince uzun ve detaylı bir cevap; Tanzimat ile ilgili.",
        ]
    )
    _install_pipeline_mocks(populated_retriever, mock_llm, monkeypatch)

    pipeline = build_pipeline()
    result = await pipeline.ainvoke(_initial_state())

    assert result["attempts"] == 2, f"İki attempt beklenir, gelen: {result['attempts']}"
    assert "Yeterince uzun" in result["answer"], "Final cevap uzun olanı olmalı"
    assert result["needs_retry"] is False, "İkinci cevap geçmeli"
    assert mock_llm.call_count == 2


@pytest.mark.asyncio
async def test_pipeline_max_attempts(
    populated_retriever: EduRetriever,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mock LLM hep kısa cevap → 3 deneme sonra zorla biter (sonsuz retry yok)."""
    # 5 kısa cevap; sadece 3'ü tüketilmeli.
    mock_llm = MockLLM(["x"] * 5)
    _install_pipeline_mocks(populated_retriever, mock_llm, monkeypatch)

    pipeline = build_pipeline()
    result = await pipeline.ainvoke(_initial_state())

    # Validator logic: attempts < _MAX_ATTEMPTS (3) AND _is_weak_answer.
    # attempt 1: weak → retry. attempt 2: weak → retry. attempt 3: validator
    # attempts<3 False → no retry → format. Yani 3 LLM call.
    assert result["attempts"] == 3, (
        f"Max 3 attempt beklenir, gelen: {result['attempts']}"
    )
    assert result["needs_retry"] is False, "Max attempts'ta retry kapalı"
    assert mock_llm.call_count == 3, "LLM 3 kez çağrılmalı (max attempts)"
    # Cevap hâlâ kısa ama format'a geçti, sources doldu.
    assert result["sources"], "Sources format_node tarafından doldurulmuş olmalı"
