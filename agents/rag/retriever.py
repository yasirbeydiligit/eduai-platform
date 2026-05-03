"""RAG retriever — query → Qdrant similarity search → LangChain Document'lar.

SPEC `EduRetriever` imzası:
    retrieve(query, subject=None, k=4) -> list[Document]
    get_context_string(docs) -> str

Tasarım kararları (Sapma 12-15 — IMPLEMENTATION_NOTES'a eklenecek):
- **Document tipi**: `langchain_core.documents.Document` — LangGraph
  retrieve_node'u (Task 3) doğrudan tüketebilsin, CrewAI tool'u (Task 4)
  da aynı tipi paylaşsın.
- **Score**: SPEC dönüş tipi sadece `list[Document]`; biz **score'u
  `metadata["score"]`'da saklıyoruz**. Validator node (Task 3) eşik bazlı
  retry için bu skoru okuyacak.
- **Sync API**: SPEC `nodes.py` örneğinde `await retriever.retrieve(...)`
  yazıyor ama qdrant-client sync; LangGraph node async'i içinde sync
  fonksiyon çağırılabilir → retrieve sync, nodes.py'da `await` yok
  (Task 3'te uygulanacak).
- **Subject filter**: payload `subject == <value>` filter'ı; subject
  None ise tüm collection'dan ara.
- **Context formatı**: numaralı + kaynak/sayfa header'lı — model'in
  alıntı yapması için netlik (hangi chunk hangi dosyadan).
"""

from __future__ import annotations

import os

import structlog
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from agents.rag.embeddings import TurkishEmbedder
from agents.rag.indexer import DEFAULT_COLLECTION

logger = structlog.get_logger(__name__)


class EduRetriever:
    """SPEC EduRetriever + score metadata + Türkçe context formatı."""

    def __init__(
        self,
        qdrant_url: str | None = None,
        collection_name: str | None = None,
        embedder: TurkishEmbedder | None = None,
        client: QdrantClient | None = None,
    ) -> None:
        """Retriever kur. Qdrant connection aç, embedder DI veya default.

        Args:
            qdrant_url: Override; None ise ENV `QDRANT_URL` veya localhost.
            collection_name: Override; None ise ENV `QDRANT_COLLECTION` veya default.
            embedder: DI — yoksa TurkishEmbedder() (e5-large default).
            client: Qdrant client DI (Sapma 33). Test'te in-memory.
        """
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.collection_name = collection_name or os.getenv(
            "QDRANT_COLLECTION", DEFAULT_COLLECTION
        )
        self.embedder = embedder or TurkishEmbedder()
        # timeout=10: query_points yanıtı kısa olmalı; uzun sürerse Qdrant
        # arızası sinyali, hızlı düşelim.
        self.client = (
            client
            if client is not None
            else QdrantClient(url=self.qdrant_url, timeout=10.0)
        )

        logger.debug(
            "retriever_ready",
            qdrant_url=self.qdrant_url,
            collection=self.collection_name,
            embedder_model=self.embedder.model_id,
        )

    def retrieve(
        self,
        query: str,
        subject: str | None = None,
        k: int = 4,
    ) -> list[Document]:
        """Sorguya en yakın k chunk'ı döndür.

        Args:
            query: Türkçe kullanıcı sorusu.
            subject: Verilirse sadece o ders metadata'sıyla filtreler.
            k: Kaç chunk döndür (default 4 — RAG context için tipik).

        Returns:
            LangChain Document listesi. Her Document'ta:
              - page_content: chunk metni
              - metadata.source: dosya adı
              - metadata.page: PDF sayfası (TXT için 0)
              - metadata.subject: ders adı (varsa)
              - metadata.chunk_index: doküman içi sıra
              - metadata.doc_id: dedup anahtarı
              - metadata.score: cosine similarity (validator için)
            Sıra: skor azalan.
        """
        # Query embed — E5'te "query: " prefix otomatik uygulanır.
        q_vec = self.embedder.embed_query(query)

        # Subject filter — payload'da subject eşleşmesi zorunlu.
        # Yoksa filter=None tüm collection.
        query_filter: qmodels.Filter | None = None
        if subject:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="subject",
                        match=qmodels.MatchValue(value=subject),
                    )
                ]
            )

        # query_points: qdrant-client v1.10+ tercih edilen API; deprecated
        # `search` yerine. with_payload=True → metadata'yı geri al.
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=q_vec,
            query_filter=query_filter,
            limit=k,
            with_payload=True,
        )

        docs: list[Document] = []
        for point in result.points:
            payload = point.payload or {}
            # page_content: chunk metni — indexer payload'da "text" anahtarı altında.
            # Boşsa Document yaratma (anlamsız).
            content = payload.get("text", "")
            if not content:
                continue
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": payload.get("source"),
                        "page": payload.get("page_num"),
                        "subject": payload.get("subject"),
                        "chunk_index": payload.get("chunk_index"),
                        "doc_id": payload.get("doc_id"),
                        # Score validator threshold için kritik (Task 3 LangGraph).
                        "score": float(point.score),
                    },
                )
            )

        logger.info(
            "retrieved",
            query_length=len(query),
            subject=subject,
            k=k,
            results=len(docs),
            top_score=docs[0].metadata["score"] if docs else None,
        )
        return docs

    @staticmethod
    def get_context_string(docs: list[Document]) -> str:
        """Chunk'ları LLM prompt'una uygun tek string'e birleştir.

        Format:
            [1] (kaynak: <dosya>, sayfa: <n>)
            <chunk metni>

            [2] (kaynak: ...)
            ...

        Numaralandırma model'in alıntı yapmasını kolaylaştırır
        ("[1]'de söylendiği gibi..."); kaynak/sayfa header'ı validator'ın
        kaynak listesi çıkarmasında işe yarar.
        """
        if not docs:
            return ""

        parts: list[str] = []
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "<bilinmiyor>")
            page = doc.metadata.get("page", 0)
            parts.append(f"[{i}] (kaynak: {source}, sayfa: {page})\n{doc.page_content}")
        return "\n\n".join(parts)
