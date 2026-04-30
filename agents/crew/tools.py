"""CrewAI agent tool'ları — RAG retriever köprüsü.

`search_education_materials` agent'ların RAG retriever'a erişmesini sağlar.
TASKS.md tanımına uyar; ek olarak F-2 (CrewAI 1.x) güncel API'sini kullanır
(`crewai.tools.tool` decorator'ı).

Tasarım: CrewAI agent'ı tool'u doğal dilden çağırırken iki argüman geçer
(query + subject). Tool string döndürür → LLM context'te kullanır.
"""

from __future__ import annotations

from crewai.tools import tool

from agents.rag.retriever import EduRetriever


@tool("search_education_materials")
def search_education_materials(query: str, subject: str = "") -> str:
    """Türkçe eğitim materyallerinde semantic search yapar.

    Researcher agent bu tool'u çağırarak Qdrant'taki ders kitabı
    chunk'larından ilgili pasajları çeker. Sonuç numaralı + kaynak
    header'lı string formatında döner — Writer agent context olarak
    kullanır.

    Args:
        query: Aranacak konu veya soru (Türkçe).
        subject: Opsiyonel ders filtresi ("tarih", "fizik" vb.).
                 Boş bırakılırsa tüm collection'da arar.

    Returns:
        Numaralı + kaynaklı bağlam string'i; sonuç yoksa açıklayıcı mesaj.
    """
    retriever = EduRetriever()
    # CrewAI argüman olarak boş string geçirebilir; subject="" → None.
    subj = subject.strip() or None
    docs = retriever.retrieve(query=query, subject=subj, k=4)

    if not docs:
        return (
            f"'{query}' sorgusu için "
            f"{f'{subj} dersinde ' if subj else ''}"
            "ilgili materyal bulunamadı."
        )

    return retriever.get_context_string(docs)
