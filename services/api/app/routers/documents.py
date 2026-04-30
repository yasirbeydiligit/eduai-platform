"""Döküman upload endpoint'i — multipart/form-data + dosya validasyonu.

Validasyon router'da (HTTP sınırı); iş mantığı (metadata kayıt) DocumentService'de.

P3 Task 5: upload sonrası DocumentIndexer.index_file() çağrısı eklendi
(Sapma 29: tempfile pattern). chunks_indexed response'a yansır.
"""

import tempfile
from pathlib import Path
from typing import Annotated

import structlog

# Runtime import — FastAPI Depends() introspection için zorunlu
# (TYPE_CHECKING'de tutulursa indexer parametresi query param olarak yorumlanır).
from agents.rag.indexer import DocumentIndexer
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.dependencies import get_document_indexer, get_document_service
from app.schemas.documents import DocumentUploadResponse
from app.schemas.questions import SubjectEnum
from app.services.document_service import DocumentService

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/v1/documents", tags=["Documents"])

# Upload sınırları — magic number yerine named constant. P1'de hardcoded,
# ileride settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024 ile parametrik yapılabilir.
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS: tuple[str, ...] = (".pdf", ".txt")


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Döküman yükle",
    description=(
        "PDF veya TXT dosyası (max 10MB). Metadata kaydedilir + Qdrant'a "
        "chunk'lara bölünerek indekslenir. Aynı içerik tekrar yüklenirse "
        "duplicate-skip (chunks_indexed=0)."
    ),
)
async def upload_document(
    # Annotated[T, File/Form/Depends(...)] modern FastAPI pattern — ruff B008 uyumu.
    # UploadFile streaming destekli — büyük dosyaları tamamen RAM'e yüklemez.
    file: Annotated[UploadFile, File(description="PDF veya TXT dosyası (max 10MB)")],
    title: Annotated[str, Form(max_length=200, description="Döküman başlığı")],
    subject: Annotated[SubjectEnum, Form(description="Ders konusu")],
    grade_level: Annotated[int, Form(ge=1, le=12, description="Sınıf seviyesi (1-12)")],
    service: Annotated[DocumentService, Depends(get_document_service)],
    indexer: Annotated[DocumentIndexer, Depends(get_document_indexer)],
) -> DocumentUploadResponse:
    """Dosyayı valide et, metadata'yı servise kaydet, Qdrant'a indeksle ve response döndür."""
    # --- 1) Extension validasyonu ---
    # file.filename None olabilir (çok nadir), güvenli kontrol için fallback ""
    # .lower() → "FILE.PDF" gibi upper-case isimleri de kabul et
    filename = (file.filename or "").lower()
    if not filename.endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF and TXT files allowed",
        )

    # --- 2) İçeriği bir kez oku → hem boyut hem indeks için kullan ---
    # Önceki pattern (size attr) sadece validation için yeterliydi; Task 5'te
    # zaten içeriği dosyaya yazacağız → bir kez okuyup hem ölç hem yaz.
    contents = await file.read()
    file_size = len(contents)

    if file_size > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File too large",
        )

    # --- 3) Tempfile pattern (Sapma 29) — indexer Path bekliyor ---
    # NamedTemporaryFile delete=False: with bloğu çıkışında siliyoruz; aksi
    # halde indexer.index_file() bitmeden dosya silinebilir.
    suffix = Path(filename).suffix  # .pdf veya .txt
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    chunks_indexed = 0
    try:
        # Qdrant'a indeksle. doc_id content hash'inden → idempotent.
        # Aynı dosya tekrar yüklenirse chunks_indexed = 0 (duplicate-skip).
        chunks_indexed = indexer.index_file(
            tmp_path,
            metadata={"subject": subject.value, "grade_level": grade_level},
            # Orijinal filename — doc_id stem'i bundan hesaplanır (Sapma 32).
            # Tempfile path'inin rastgele stem'i doc_id'yi etkilemez → aynı
            # içerikli aynı dosya tekrar yüklenirse duplicate-skip çalışır.
            source_name=file.filename,
        )
    except Exception as exc:
        # Indeks başarısız — dosyayı temizle, kullanıcıya net hata ver.
        logger.error(
            "document_index_failed",
            filename=file.filename,
            exc_type=type(exc).__name__,
            exc_msg=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indeksleme hatası: {exc}",
        ) from exc
    finally:
        # Temp dosyayı her halükarda sil — hata olsa bile disk kirletmesin.
        tmp_path.unlink(missing_ok=True)

    # --- 4) Service'e metadata kaydı ---
    response = await service.save_document(
        title=title,
        subject=subject,
        grade_level=grade_level,
        file_size=file_size,
    )

    # Indeks sonucunu response'a yansıt. Pydantic v2 model'leri mutable;
    # response.chunks_indexed = ... daha temiz olur model_copy'den.
    response.chunks_indexed = chunks_indexed
    # status: indeks başarılıysa "ready", duplicate-skip ise yine "ready"
    # (zaten indeksli demek). 0 chunk değil 0 yeni chunk demek.
    response.status = "ready"

    logger.info(
        "document_upload_complete",
        document_id=str(response.document_id),
        chunks_indexed=chunks_indexed,
        was_duplicate=chunks_indexed == 0,
    )
    return response
