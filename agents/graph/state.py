"""LangGraph AgentState — pipeline boyunca taşınan ortak veri yapısı.

LangGraph'ın TypedDict + reducer pattern'i: her node state'in **partial
update**'ini döndürür; LangGraph delta'ları reducer'larla merge eder.
Reducer belirtilmeyen alanlar default replace (üzerine yazma).

`messages` alanı `add_messages` reducer'ı kullanır → list'e append eder
(replace etmez). Konuşma geçmişi tutmak için kritik (session_memory.py
ile entegrasyon Task 4+'da).

SPEC.md schema'sına uyar; tüm alanlar `total=False` ile opsiyonel
(initial state'te kullanıcı yalnızca question, subject, grade_level,
session_id, attempts, needs_retry verir).
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """LangGraph pipeline state.

    Initial state alanları (kullanıcı sağlar):
        question: Türkçe kullanıcı sorusu.
        subject: Filtre için ders adı (örn. "tarih"); None mümkün.
        grade_level: 9-12 lise sınıfı.
        session_id: Konuşma kimliği (memory için).
        attempts: 0'dan başlar, generate_node arttırır.
        needs_retry: validator karar verir.

    Pipeline boyunca dolan alanlar:
        retrieved_docs: RAG retriever sonuçları (LangChain Document).
        context: docs'tan üretilen tek-string prompt context'i.
        answer: LLM cevabı.
        confidence: 0-1 arası güven skoru (top retrieved doc score).
        sources: Cevapta atıf yapılan unique kaynak listesi.
        messages: add_messages reducer ile append-only sohbet geçmişi.
    """

    # --- Initial (kullanıcı verir) ---
    question: str
    subject: str | None
    grade_level: int
    session_id: str

    # --- Retrieval çıktıları ---
    retrieved_docs: list[Document]
    context: str

    # --- Generation çıktıları ---
    answer: str
    confidence: float
    sources: list[str]

    # --- Validator state ---
    attempts: int
    needs_retry: bool

    # --- Konuşma geçmişi (Task 4+ memory entegrasyonu) ---
    messages: Annotated[list, add_messages]
