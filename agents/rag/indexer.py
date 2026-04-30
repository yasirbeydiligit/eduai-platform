"""Doküman indekleyici — TXT/PDF dosyalarını chunk'layıp Qdrant'a yükler.

SPEC `DocumentIndexer` imzasına uyar; TASKS.md ekstra gereksinimleri
(duplicate detection, zengin metadata, progress, hata izleme) ile zenginleştirilmiştir.

Mimari kararlar:
- **Doc ID** = `{stem}_{content_sha256[:16]}`. Aynı dosyayı tekrar yüklemek
  no-op olur; içerik değiştiyse farklı doc_id ile re-index edilir.
- **Point ID** = `uuid5(NAMESPACE, "{doc_id}:{chunk_index}")` — deterministic,
  upsert idempotent (aynı chunk yeniden yazılır).
- **Chunking**: `RecursiveCharacterTextSplitter` (paragraf > cümle > kelime).
  chunk_size karakter cinsinden 500 (~120 Türkçe token) — Sapma 9 detay.
- **Metadata**: `source` (dosya adı), `page_num` (PDF sayfası, TXT için 0),
  `subject` (kullanıcı verir), `chunk_index` (0-based, doküman içinde),
  `doc_id` (dedup anahtarı), `text` (tam chunk içeriği — Qdrant payload).
- **Progress**: encode + upsert sonrası "X chunk yüklendi" + per-chunk hata
  yakalama (chunk index + ilk 80 char raporlanır).
"""

from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path

import structlog
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from agents.rag.embeddings import TurkishEmbedder

logger = structlog.get_logger(__name__)

# Default collection — P1 API + agents/ ortak. ENV `QDRANT_COLLECTION` ile override.
DEFAULT_COLLECTION = "eduai_documents"

# Chunk parametreleri SPEC'ten. Karakter-bazlı; Türkçe için ~120 token.
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50

# UUID5 namespace — point ID'leri deterministic ve upsert idempotent yapmak için.
# Sabit bir namespace UUID; production'da değiştirme (ID'ler bağlı).
_POINT_NAMESPACE = uuid.UUID("a3c4f1d2-7890-4abc-def0-1234567890ab")

# Türkçe metin için splitter ayırıcı hiyerarşisi: paragraf → cümle → kelime.
# Standart İngilizce default'larından farklı: "? " ve "! " sonu cümleyi
# kapsayacak şekilde eklenmiş; "; " bağlaç bölümü için.
_TURKISH_SEPARATORS: list[str] = ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]


class IndexError(Exception):
    """Indekleme sırasında oluşan hatalar (chunk-spesifik tutar)."""

    def __init__(self, chunk_index: int, chunk_preview: str, cause: Exception) -> None:
        self.chunk_index = chunk_index
        self.chunk_preview = chunk_preview
        self.cause = cause
        super().__init__(
            f"Chunk {chunk_index} indeklenemedi ({type(cause).__name__}: {cause}). "
            f"İçerik önizleme: «{chunk_preview[:80]}...»"
        )


class DocumentIndexer:
    """SPEC.md'deki DocumentIndexer + duplicate-control + progress + zengin metadata."""

    def __init__(
        self,
        qdrant_url: str | None = None,
        collection_name: str | None = None,
        embedder: TurkishEmbedder | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        """Indexer kur. Qdrant connection aç, collection yoksa oluştur.

        Args:
            qdrant_url: Override; None ise ENV `QDRANT_URL` veya localhost.
            collection_name: Override; None ise ENV `QDRANT_COLLECTION` veya default.
            embedder: DI — yoksa default TurkishEmbedder() (e5-large).
            chunk_size: Karakter cinsinden hedef chunk büyüklüğü.
            chunk_overlap: Komşu chunk'lar arası örtüşme (bağlam kaybı önler).
        """
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.collection_name = collection_name or os.getenv(
            "QDRANT_COLLECTION", DEFAULT_COLLECTION
        )
        self.embedder = embedder or TurkishEmbedder()
        # timeout=30: büyük dosya upsert'lerde Qdrant ack uzayabilir.
        self.client = QdrantClient(url=self.qdrant_url, timeout=30.0)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=_TURKISH_SEPARATORS,
            # Default token sayım yerine karakter bazlı (length_function=len OK).
        )
        self._ensure_collection()
        logger.info(
            "indexer_ready",
            qdrant_url=self.qdrant_url,
            collection=self.collection_name,
            embedder_model=self.embedder.model_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    # ----- Collection yönetimi -----

    def _ensure_collection(self) -> None:
        """Collection yoksa oluştur; varsa vector_size uyumunu doğrula.

        Raises:
            RuntimeError: Mevcut collection farklı vector_size ile kurulmuşsa.
        """
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name in existing:
            info = self.client.get_collection(self.collection_name)
            # vectors_config tek vektör için VectorParams; multi-vector setup'larda dict.
            existing_size = info.config.params.vectors.size
            if existing_size != self.embedder.vector_size:
                raise RuntimeError(
                    f"Collection '{self.collection_name}' vector_size={existing_size}, "
                    f"embedder vector_size={self.embedder.vector_size}. "
                    f"Schema uyumsuz — collection silip yeniden oluştur veya "
                    f"farklı collection adı kullan."
                )
            logger.debug(
                "collection_exists",
                name=self.collection_name,
                vector_size=existing_size,
            )
            return

        # Yeni collection — vector_size embedder'dan, distance COSINE.
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(
                size=self.embedder.vector_size,
                distance=qmodels.Distance.COSINE,
            ),
        )
        # Doc-level filter index — list_documents() ve duplicate check hızlanır.
        # Qdrant 1.10+ keyword index payload field'ı için scroll filter'ı O(log n).
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="doc_id",
            field_schema=qmodels.PayloadSchemaType.KEYWORD,
        )
        logger.info(
            "collection_created",
            name=self.collection_name,
            vector_size=self.embedder.vector_size,
        )

    # ----- Doc ID + duplicate kontrol -----

    @staticmethod
    def _compute_doc_id(stem: str, content: str) -> str:
        """İçerik hash'inden deterministic doc_id üret.

        Args:
            stem: Doküman tanımlayıcı stem (uzantısız ad). API tempfile
                  ile çağrıldığında orijinal filename'in stem'i geçilir
                  → tempfile rastgele isimleri doc_id'yi etkilemez.
            content: Tam dosya içeriği (hash için).

        Aynı stem + aynı içerik → aynı doc_id → duplicate skip.
        İçerik değiştirilirse hash farklı → re-index (eski chunk'lar
        cleanup edilmez, kullanıcının kararı; bu MVP).
        """
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        return f"{stem}_{digest}"

    def _document_exists(self, doc_id: str) -> bool:
        """Doc_id ile en az 1 nokta var mı? scroll(limit=1) hızlı yoklama."""
        result, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="doc_id", match=qmodels.MatchValue(value=doc_id)
                    )
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return len(result) > 0

    # ----- Dosya okuma -----

    @staticmethod
    def _read_file(file_path: Path) -> list[tuple[str, int]]:
        """Dosyayı oku → liste halinde (metin, page_num) parçaları döndür.

        - TXT: tek bir (full_text, 0) tuple.
        - PDF: her sayfa ayrı tuple (page_num 1-based).

        Raises:
            ValueError: Desteklenmeyen format.
            FileNotFoundError: Dosya yoksa.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix == ".txt":
            return [(file_path.read_text(encoding="utf-8"), 0)]
        if suffix == ".pdf":
            reader = PdfReader(str(file_path))
            return [
                (page.extract_text() or "", i + 1)
                for i, page in enumerate(reader.pages)
            ]
        raise ValueError(f"Desteklenmeyen dosya formatı: {suffix} ({file_path.name})")

    # ----- Ana indeks akışı -----

    def index_file(
        self,
        file_path: str | Path,
        metadata: dict | None = None,
        source_name: str | None = None,
    ) -> int:
        """Tek bir dosyayı indeks pipeline'ından geçir.

        Args:
            file_path: TXT veya PDF dosya yolu (gerçek disk konumu).
            metadata: subject (zorunlu — filtre için), grade_level vs. opsiyonel.
                      Otomatik metadata: source, page_num, chunk_index, doc_id.
            source_name: Orijinal dosya adı (örn. "tarih_tanzimat.txt"). API
                         tempfile pattern'inde kritik (Sapma 32) — file_path
                         tempfile rastgele yolu olur ama doc_id orijinal
                         filename'den hesaplanmalı, aksi halde duplicate-skip
                         çalışmaz. None ise file_path.name kullanılır
                         (lokal script senaryosu).

        Returns:
            Yüklenen chunk sayısı (duplicate skip durumunda 0).

        Raises:
            ValueError: Desteklenmeyen format veya boş içerik.
            IndexError: Embedding/upsert sırasında bir chunk hata verirse;
                        hangi chunk + neden bilgisiyle.
        """
        file_path = Path(file_path)
        metadata = dict(metadata or {})

        # Tüm dosya içeriğini birleştir (PDF için tüm sayfalar concat) →
        # doc_id hesaplaması için tek bir hash; sayfa parça sayfa indeks
        # akışında zaten ayrı işlenir.
        pages = self._read_file(file_path)
        full_content = "\n".join(text for text, _ in pages if text.strip())
        if not full_content.strip():
            raise ValueError(f"Boş dosya içeriği: {file_path}")

        # Stem ve display name — source_name varsa orijinal, yoksa fallback.
        # source_name "tarih_tanzimat.txt" gibi tam isim; .stem = "tarih_tanzimat".
        display_name = source_name or file_path.name
        stem_for_id = Path(display_name).stem
        doc_id = self._compute_doc_id(stem_for_id, full_content)

        # Duplicate kontrolü — aynı dosya tekrar yüklenirse early return.
        if self._document_exists(doc_id):
            print(f"  ⏭  Atlanıyor — '{display_name}' zaten indeksli (doc_id={doc_id})")
            logger.info(
                "document_already_indexed",
                file=str(file_path),
                source=display_name,
                doc_id=doc_id,
            )
            return 0

        print(f"  📄 İndeksleniyor: {display_name} (doc_id={doc_id})")

        # Sayfa bazlı chunking → her chunk page_num metadata'sını korur.
        # PDF'te chunk sınırı sayfa içinde kalır (sayfa-arası bağlam kaybı kabul).
        chunks: list[tuple[str, int]] = []  # (chunk_text, page_num)
        for page_text, page_num in pages:
            if not page_text.strip():
                continue
            for piece in self.splitter.split_text(page_text):
                if piece.strip():
                    chunks.append((piece, page_num))

        if not chunks:
            raise ValueError(f"Splitter chunk üretmedi: {file_path}")

        total = len(chunks)
        print(f"  ✂️  {total} chunk üretildi, embedding başlıyor...")

        # Toplu encode — embedder MPS/CUDA'da batch hesabı verimli yapar.
        try:
            chunk_texts = [c for c, _ in chunks]
            vectors = self.embedder.embed_documents(chunk_texts)
        except Exception as exc:
            # Toplu encode başarısız — hangisinin hata verdiğini izole et.
            for i, text in enumerate(chunk_texts):
                try:
                    self.embedder.embed_query(text)  # tek tek dene
                except Exception as inner:
                    raise IndexError(i, text, inner) from inner
            # Tek tek de geçerse genel hata orijinaldir; up edelim.
            raise IndexError(-1, "(toplu encode)", exc) from exc

        # Qdrant points hazırla — point ID deterministic.
        points: list[qmodels.PointStruct] = []
        for chunk_index, ((chunk_text, page_num), vector) in enumerate(
            zip(chunks, vectors)
        ):
            payload = {
                **metadata,  # subject, grade_level vb. kullanıcı metadata
                "source": display_name,  # orijinal filename (Sapma 32)
                "page_num": page_num,
                "chunk_index": chunk_index,
                "doc_id": doc_id,
                "text": chunk_text,  # validator/UI için chunk içeriği erişilebilir
            }
            point_id = str(uuid.uuid5(_POINT_NAMESPACE, f"{doc_id}:{chunk_index}"))
            points.append(
                qmodels.PointStruct(id=point_id, vector=vector, payload=payload)
            )

        # Upsert — wait=True: ack edilene kadar bloke (test'te eventual
        # consistency sürprizi olmaz).
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
        except Exception as exc:
            # Upsert hata verince hangi chunk index'inin failed olduğunu net söyleme
            # (Qdrant batch hatası tüm batch'i drop eder); chunk_preview ilki olur.
            preview = chunks[0][0] if chunks else ""
            raise IndexError(0, preview, exc) from exc

        print(f"  ✓ {total} chunk yüklendi (collection='{self.collection_name}')")
        logger.info(
            "document_indexed",
            file=str(file_path),
            doc_id=doc_id,
            chunks=total,
            collection=self.collection_name,
        )
        return total

    # ----- Listeleme -----

    def list_documents(self) -> list[dict]:
        """Collection'da yüklü doküman özetlerini döndür.

        Returns:
            Her doküman için: doc_id, source, subject, page_count, chunk_count.
            Aynı doc_id'nin tüm chunk'ları gruplanır.
        """
        # Scroll: tüm point'leri sayfa sayfa çek; payload yeter, vector lazım değil.
        # Büyük collection'larda (>10k point) bu pahalı — production'da aggregate
        # sorgu Qdrant 1.10+ ile mümkün, MVP için scroll yeterli.
        accumulator: dict[str, dict] = {}
        next_offset = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                payload = p.payload or {}
                doc_id = payload.get("doc_id", "<bilinmiyor>")
                if doc_id not in accumulator:
                    accumulator[doc_id] = {
                        "doc_id": doc_id,
                        "source": payload.get("source"),
                        "subject": payload.get("subject"),
                        "pages": set(),
                        "chunks": 0,
                    }
                accumulator[doc_id]["pages"].add(payload.get("page_num", 0))
                accumulator[doc_id]["chunks"] += 1
            if next_offset is None:
                break

        # set → sorted list (JSON serializable)
        return [
            {
                **entry,
                "pages": sorted(entry["pages"]),
                "page_count": len(entry["pages"]),
            }
            for entry in accumulator.values()
        ]
