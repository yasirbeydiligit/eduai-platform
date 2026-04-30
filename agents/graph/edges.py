"""LangGraph kenar (edge) mantığı — conditional routing fonksiyonları.

LangGraph node'ları sequential `add_edge` ile bağlanır; karar gerektiren
yerlerde `add_conditional_edges` + bu modüldeki router fonksiyonları
kullanılır. State alır → bir sonraki node adını döndürür.
"""

from __future__ import annotations

from agents.graph.state import AgentState


def needs_retry_router(state: AgentState) -> str:
    """validate_node sonrası: retry mi format mı?

    Returns:
        "generate" — validator zayıf bulduysa generate'e geri (retry).
        "format"   — cevap onaylandı veya max attempts; format'a git.
    """
    return "generate" if state.get("needs_retry") else "format"
