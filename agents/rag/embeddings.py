"""Türkçe metin embedding modeli wrapper'ı.

P3 RAG pipeline'ında corpus chunk'larını ve user query'lerini vektöre çeviren
tek katman. SPEC'teki `TurkishEmbedder` imzasına uyar; ek olarak:

- F-1 Sapma 7 kararı (docs/p3/IMPLEMENTATION_NOTES.md): default model
  `intfloat/multilingual-e5-large` (zor benchmark'ta avg score 0.86, geniş
  margin). `EMBEDDING_MODEL` ENV var ile config-driven değiştirilebilir.
- E5 ailesi için instruction prefix (`query: ` / `passage: `) otomatik uygulanır;
  diğer modellerde no-op. Bu retrieval kalitesini ek artırır.
- Lazy model load: ilk encode çağrısında yükleme. Test/CI'da import zamanı
  ağırlık olmaz.
- Device auto: MPS (Apple Silicon) > CUDA > CPU.

Kullanım:
    embedder = TurkishEmbedder()             # default e5-large
    vec = embedder.embed_query("Tanzimat ne zaman ilan edildi?")
    vecs = embedder.embed_documents(["paragraf 1", "paragraf 2"])
    print(embedder.vector_size)              # Qdrant collection schema için
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    # SentenceTransformer import'u runtime'da (lazy load) yapılır; type
    # checker için sadece bu blokta görünür → import maliyeti import zamanı
    # bedava.
    from sentence_transformers import SentenceTransformer

logger = structlog.get_logger(__name__)

# F-1 sapma 7: default model. E5-large zor distractor benchmark'ında
# avg score 0.86, %100 top-1.
DEFAULT_MODEL_ID = "intfloat/multilingual-e5-large"

# Instruction prefix kuralları. E5 ailesi query/passage ayrımı zorunlu;
# diğer modeller no-op. Yeni model eklendiğinde buraya kayıt yeter.
# Substring match: "e5" geçen tüm E5 türevleri (multilingual, base, large vs).
_PREFIX_RULES: tuple[tuple[str, str, str], ...] = (
    # (substring_match, query_prefix, passage_prefix)
    ("e5", "query: ", "passage: "),
)


def _resolve_prefixes(model_id: str) -> tuple[str, str]:
    """Model adına göre query/passage prefix çiftini bul.

    Returns:
        (query_prefix, passage_prefix); model bilinmiyorsa ("", "").
    """
    lowered = model_id.lower()
    for substr, qp, pp in _PREFIX_RULES:
        if substr in lowered:
            return qp, pp
    return "", ""


def _resolve_device(explicit: str | None = None) -> str:
    """Çalıştırma cihazını seç. MPS (M-series Mac) > CUDA > CPU.

    Args:
        explicit: Kullanıcı override (örn. test'te "cpu" zorlama).

    Returns:
        torch device string.
    """
    if explicit is not None:
        return explicit
    # torch import'u burada — sentence-transformers zaten torch çekiyor,
    # ek maliyet yok ama lazy load ilkesini koruyalım.
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class TurkishEmbedder:
    """Türkçe metin embedding wrapper'ı (SPEC + F-1 sapma 7).

    Attributes:
        model_id: Yüklenen sentence-transformers modeli kimliği.
        device: torch device string (mps/cuda/cpu).
        query_prefix / passage_prefix: encode öncesi metne eklenen önek.
    """

    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
    ) -> None:
        """Embedder oluştur. Model lazy yüklenir; __init__ ucuz.

        Args:
            model_id: Override; None ise ENV `EMBEDDING_MODEL` veya default.
            device: Override; None ise auto-detect.
        """
        # Çözüm sırası: arg > ENV > default. ENV var ile production swap kolay.
        self.model_id: str = model_id or os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL_ID)
        self.device: str = _resolve_device(device)
        self.query_prefix, self.passage_prefix = _resolve_prefixes(self.model_id)
        # Lazy slot — ilk encode call'da doldurulur.
        self._model: SentenceTransformer | None = None
        self._vector_size: int | None = None

        logger.debug(
            "embedder_initialized",
            model_id=self.model_id,
            device=self.device,
            query_prefix=self.query_prefix,
            passage_prefix=self.passage_prefix,
        )

    def _ensure_loaded(self) -> SentenceTransformer:
        """Model henüz yüklenmediyse yükle. Idempotent."""
        if self._model is not None:
            return self._model

        # Import burada — modülün import zamanı sentence-transformers + torch
        # ağırlığını çekmesin diye lazy.
        from sentence_transformers import SentenceTransformer

        t0 = time.perf_counter()
        # trust_remote_code=False: native ST uyumlu modellerimizde (e5-large,
        # emrecan) custom code yok; güvenlik için False.
        # gte-multilingual-base trust_remote_code=True istiyordu ama F-1 sapma
        # 7'de elendi (ABI çakışması).
        self._model = SentenceTransformer(
            self.model_id,
            device=self.device,
            trust_remote_code=False,
        )
        # ST 5.x rename + 4.x fallback (benchmark scriptindeki pattern).
        if hasattr(self._model, "get_embedding_dimension"):
            self._vector_size = self._model.get_embedding_dimension()
        else:
            self._vector_size = self._model.get_sentence_embedding_dimension()
        load_time = time.perf_counter() - t0

        logger.info(
            "embedder_loaded",
            model_id=self.model_id,
            device=self.device,
            vector_size=self._vector_size,
            load_seconds=round(load_time, 2),
        )
        return self._model

    @property
    def vector_size(self) -> int:
        """Embedding boyutu (Qdrant collection schema için).

        Erişim model'i yükler — Qdrant collection oluşturma zamanı zaten
        modelin hazır olması gerek.
        """
        self._ensure_loaded()
        # _ensure_loaded sonrası _vector_size garanti dolu.
        assert self._vector_size is not None
        return self._vector_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Bir doküman listesini vektörleştir (passage semantiği).

        Args:
            texts: Encode edilecek string'ler (boş liste → boş çıktı).

        Returns:
            Her metin için L2-normalize edilmiş float listesi (cosine için
            uygun). Sırası girdi sırasıyla aynı.
        """
        if not texts:
            return []

        model = self._ensure_loaded()
        # Passage prefix: E5'te "passage: <metin>"; diğer modellerde no-op.
        prefixed = [self.passage_prefix + t for t in texts]

        t0 = time.perf_counter()
        # convert_to_numpy=True → numpy array; tolist() ile SPEC imzasına uyar.
        # show_progress_bar=False → kütüphane bağlamında gürültü yapma;
        # progress UI indexer.py'da kullanıcıya özel print pattern'i ile.
        embeddings = model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        encode_time = time.perf_counter() - t0

        logger.debug(
            "documents_embedded",
            count=len(texts),
            seconds=round(encode_time, 3),
            ms_per_doc=round(encode_time / len(texts) * 1000, 1),
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Tek bir kullanıcı sorgusunu vektörleştir (query semantiği).

        E5'te 'query: ' prefix'i passage encoding'inden farklı bir embedding
        uzayına götürür; bu retrieval doğruluğunu artırır.

        Args:
            query: Kullanıcı sorgusu (Türkçe).

        Returns:
            L2-normalize edilmiş float listesi.
        """
        model = self._ensure_loaded()
        prefixed = self.query_prefix + query

        t0 = time.perf_counter()
        embedding = model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        encode_time = time.perf_counter() - t0

        logger.debug(
            "query_embedded",
            query_length=len(query),
            seconds=round(encode_time, 3),
        )
        return embedding.tolist()


if __name__ == "__main__":
    # Hızlı smoke test — modülü doğrudan çalıştırınca çalışır.
    # `python -m agents.rag.embeddings` veya
    # `python agents/rag/embeddings.py` ile.
    embedder = TurkishEmbedder()
    print(f"Model: {embedder.model_id}")
    print(f"Device: {embedder.device}")
    print(
        f"Prefixes: query={embedder.query_prefix!r} passage={embedder.passage_prefix!r}"
    )
    print(f"Vector size: {embedder.vector_size}")

    # Mini retrieval testi: 1 sorgu vs 2 paragraf — beklenen yakın olan kazanır.
    docs = [
        "Tanzimat Fermanı 1839'da Sultan Abdülmecid döneminde ilan edildi.",
        "Newton'un üçüncü yasası etki-tepki ilkesini açıklar.",
    ]
    query = "1839'da hangi belge ilan edildi?"

    doc_vecs = embedder.embed_documents(docs)
    q_vec = embedder.embed_query(query)

    # Cosine sim (normalize edildi, dot product yeter).
    sims = [sum(qi * di for qi, di in zip(q_vec, dv)) for dv in doc_vecs]
    print(f"\nQuery: {query}")
    for i, (doc, sim) in enumerate(zip(docs, sims)):
        print(f"  [{i}] sim={sim:.3f}  «{doc[:60]}...»")
    best = sims.index(max(sims))
    print(f"  → En yakın: doc[{best}] (beklenen: 0)")
    assert best == 0, "Smoke test FAILED — sorgu Tanzimat paragrafına yakın değil!"
    print("\n✓ Smoke test PASSED")
