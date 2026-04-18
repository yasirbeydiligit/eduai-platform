"""Döküman upload endpoint'i — multipart/form-data + dosya validasyonu.

Validasyon router'da (HTTP sınırı); iş mantığı (metadata kayıt) DocumentService'de.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.dependencies import get_document_service
from app.schemas.documents import DocumentUploadResponse
from app.schemas.questions import SubjectEnum
from app.services.document_service import DocumentService

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
    description="PDF veya TXT dosyası (max 10MB). P1'de sadece metadata kaydedilir.",
)
async def upload_document(
    # UploadFile streaming destekli — büyük dosyaları tamamen RAM'e yüklemez
    file: UploadFile = File(..., description="PDF veya TXT dosyası (max 10MB)"),
    title: str = Form(..., max_length=200, description="Döküman başlığı"),
    subject: SubjectEnum = Form(..., description="Ders konusu"),
    grade_level: int = Form(..., ge=1, le=12, description="Sınıf seviyesi (1-12)"),
    service: DocumentService = Depends(get_document_service),
) -> DocumentUploadResponse:
    """Dosyayı valide et, metadata'yı servise kaydet ve response döndür."""
    # --- 1) Extension validasyonu ---
    # file.filename None olabilir (çok nadir), güvenli kontrol için fallback ""
    # .lower() → "FILE.PDF" gibi upper-case isimleri de kabul et
    filename = (file.filename or "").lower()
    if not filename.endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF and TXT files allowed",
        )

    # --- 2) Boyut validasyonu ---
    # UploadFile.size Starlette 0.21+ ile geliyor; content-length header'dan gelir.
    # Her client göndermeyebilir → None ise fallback: dosyayı oku ve ölç.
    if file.size is not None:
        file_size = file.size
    else:
        # Fallback: içeriği bellek'e al. P1 için kabul — max 10MB'lık sınır zaten var
        # ve chunked streaming validation kurmak için henüz gerekçe yok.
        contents = await file.read()
        file_size = len(contents)
        # Pointer'ı başa al — ileride dosya tekrar okunacaksa (P3'te vektörleme) bozulmasın
        await file.seek(0)

    if file_size > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File too large",
        )

    # --- 3) Service'e delege et ---
    return await service.save_document(
        title=title,
        subject=subject,
        grade_level=grade_level,
        file_size=file_size,
    )
