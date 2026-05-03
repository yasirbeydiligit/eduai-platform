"""CrewAI Writer çıktı post-validator'ları (Sapma 37 — Sapma 24 follow-up).

Sapma 24'te Writer "Kaynaklar" hallucination'ı (uydurma kitap referansları)
prompt-level fix ile çözüldü. Bu modül **production güvenlik katmanı**:
prompt fix yetmezse cevabı döndürmeden önce regex + set-difference ile
kontrol et, uyarı listesi üret.

İki seviye check:
1. **Akademik referans pattern'i** — "Newton, I. (1687)" gibi yazar (yıl)
   formatları. Bağlamımızda akademik yayın yok → herhangi bir match
   hallucination sinyali.
2. **Source whitelist** — "Kaynaklar" altında belirtilen dosya adları
   bağlamdaki retrieved sources'a ⊆ olmalı; aksi halde uydurma kaynak.

Hard fail değil — `ValidationResult` döner. Production'da:
- `is_clean=False` → log + metric increment (Prometheus counter vs)
- Aşırı uydurma → response'a "uyarı" alanı ekle veya retry tetikle
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger(__name__)

# Akademik referans pattern'i: "Yazar, A. (1234)" — Writer'ın yaptığı tipik
# uydurma. CrewAI 1.x backstory etkisiyle nadiren oluşur (Sapma 24 prompt
# fix sonrası testlerde sıfır), ama MVP defensive kontrol.
_ACADEMIC_REF_PATTERN = re.compile(
    r"\b[A-ZÇĞİÖŞÜ][a-zçğıöşü]+(?:,\s+[A-ZÇĞİÖŞÜ]\.)?\s+\(\d{4}\)"
)

# "Kaynaklar" başlığını yakala (markdown başlık, kalın, alt başlık varyantları).
# Multi-line: başlığı bul, sonraki paragrafı parse et.
_KAYNAKLAR_HEADER_PATTERN = re.compile(
    r"(?im)^\s*#{1,4}\s*Kaynaklar?\s*:?\s*$|^\s*\*\*Kaynaklar?\*\*\s*:?\s*$"
)

# Dosya adı pattern'i — payload metadata'sındaki source field'ıyla eşleşir.
_FILENAME_PATTERN = re.compile(r"\b([\w\-]+\.(?:txt|pdf|md|docx?))\b", re.IGNORECASE)


@dataclass
class ValidationResult:
    """Post-validator çıktısı.

    Attributes:
        is_clean: Hiç uyarı yoksa True.
        academic_refs: Bağlamda olmayan akademik referans pattern'leri.
        unknown_sources: "Kaynaklar"da bağlama göre tanımsız dosya adları.
        warnings: İnsan-okur uyarı mesajları (log + UI için).
    """

    is_clean: bool
    academic_refs: list[str] = field(default_factory=list)
    unknown_sources: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_writer_output(
    answer_text: str,
    allowed_sources: set[str] | list[str],
) -> ValidationResult:
    """Writer cevabını post-validate et.

    Args:
        answer_text: Writer agent'ın final markdown cevabı.
        allowed_sources: Researcher'ın RAG tool'undan elde ettiği orijinal
                         dosya adları (örn. {"tarih_tanzimat.txt", "fizik_newton.txt"}).
                         Cevapta bunların dışında bir "kaynak" varsa uydurma sayılır.

    Returns:
        ValidationResult — is_clean True ise sıfır uyarı.
    """
    allowed = {s.lower() for s in allowed_sources}
    warnings: list[str] = []

    # 1) Akademik referans pattern'i — herhangi bir match şüpheli.
    # Türkçe Tanzimat/Mustafa gibi isimler de match olabilir (false positive),
    # bu yüzden sadece "(YYYY)" formatlı (akademik tipik) olanı filtrele.
    academic_matches = _ACADEMIC_REF_PATTERN.findall(answer_text)
    if academic_matches:
        warnings.append(
            f"⚠ Akademik referans pattern'i tespit edildi ({len(academic_matches)} adet): "
            f"{academic_matches[:3]}{'...' if len(academic_matches) > 3 else ''}. "
            "Bağlamda akademik yayın yok; bu hallucination olabilir."
        )

    # 2) "Kaynaklar" bölümü parse — bağlamda olmayan dosya adı/kitap satırları.
    unknown_sources: list[str] = []
    header_match = _KAYNAKLAR_HEADER_PATTERN.search(answer_text)
    if header_match:
        # Başlıktan sonraki kısmı al — bir sonraki başlığa kadar veya dosya sonu.
        kaynaklar_section = answer_text[header_match.end() :]
        # Sonraki markdown başlığı varsa o yere kadar kes.
        next_header = re.search(r"(?m)^\s*#{1,6}\s+\S", kaynaklar_section)
        if next_header:
            kaynaklar_section = kaynaklar_section[: next_header.start()]

        # Her satırda dosya adı var mı? Yoksa "kaynak" ama whitelist'te değil mi?
        for line in kaynaklar_section.splitlines():
            line_stripped = line.strip().lstrip("-*•").strip()
            if not line_stripped:
                continue
            filenames_in_line = {
                f.lower() for f in _FILENAME_PATTERN.findall(line_stripped)
            }
            if not filenames_in_line:
                # Satırda dosya adı yok ama "kaynak" listelenmiş → şüpheli
                # (uydurma kitap referansı veya açıklama satırı olabilir).
                # 3+ kelimelik ve büyük harfle başlayan satırları flag'le.
                words = line_stripped.split()
                if len(words) >= 3 and any(c.isupper() for c in words[0]):
                    unknown_sources.append(line_stripped[:80])
                continue
            # Dosya adı var ama whitelist'te yok mu?
            for fname in filenames_in_line:
                if fname not in allowed:
                    unknown_sources.append(fname)

    if unknown_sources:
        warnings.append(
            f"⚠ Bağlam dışı 'kaynak' tespiti ({len(unknown_sources)} adet): "
            f"{unknown_sources[:3]}{'...' if len(unknown_sources) > 3 else ''}. "
            f"İzinli dosyalar: {sorted(allowed)}"
        )

    is_clean = not warnings
    if not is_clean:
        logger.warning(
            "writer_output_validation_warnings",
            academic_refs_count=len(academic_matches),
            unknown_sources_count=len(unknown_sources),
            allowed_sources=sorted(allowed),
        )
    else:
        logger.debug(
            "writer_output_clean",
            allowed_sources=sorted(allowed),
        )

    return ValidationResult(
        is_clean=is_clean,
        academic_refs=academic_matches,
        unknown_sources=unknown_sources,
        warnings=warnings,
    )
