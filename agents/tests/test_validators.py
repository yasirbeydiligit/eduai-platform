"""CrewAI post-validator unit testleri (Sapma 37 — Sapma 24 follow-up).

`validate_writer_output` davranışı:
- Temiz cevap → is_clean=True
- Akademik referans pattern → academic_refs dolu, is_clean=False
- Whitelist dışı dosya adı → unknown_sources dolu, is_clean=False
- "Kaynaklar" başlığı yoksa source check skip (sadece akademik pattern bakılır)
"""

from __future__ import annotations

from agents.crew.validators import validate_writer_output


# Bağlamımızdaki gerçek dosyalar — testler bu whitelist üzerinden çalışıyor.
_ALLOWED = {"tarih_tanzimat.txt", "fizik_newton.txt"}


def test_clean_answer_passes() -> None:
    """Sadece izinli kaynaklar ve akademik pattern yok → is_clean=True."""
    answer = """
# Tanzimat Fermanı

Tanzimat Fermanı 1839'da ilan edildi [1]. Önemli bir reformdu.

## Kaynaklar:
- tarih_tanzimat.txt: Osmanlı Tanzimat reformları
- fizik_newton.txt: Newton hareket yasaları
"""
    result = validate_writer_output(answer, _ALLOWED)
    assert result.is_clean
    assert result.academic_refs == []
    assert result.unknown_sources == []


def test_academic_reference_flagged() -> None:
    """Akademik (Yazar, Y. (YYYY)) pattern'i flag'lenmeli — Sapma 24 hallucination."""
    answer = """
# Tanzimat

Tanzimat 1839'da ilan edildi.

## Kaynaklar:
- Newton, I. (1687). Principia Mathematica
- tarih_tanzimat.txt
"""
    result = validate_writer_output(answer, _ALLOWED)
    assert not result.is_clean
    assert len(result.academic_refs) >= 1, "Newton (1687) pattern'i yakalanmalı"


def test_unknown_filename_flagged() -> None:
    """Whitelist dışı dosya adı 'Kaynaklar' altında olursa flag'lenmeli."""
    answer = """
# Newton Yasaları

## Kaynaklar:
- fizik_newton.txt: Newton hareket yasaları
- uydurma_kaynak.txt: Bu dosya bağlamda yok
"""
    result = validate_writer_output(answer, _ALLOWED)
    assert not result.is_clean
    assert "uydurma_kaynak.txt" in result.unknown_sources


def test_no_kaynaklar_section_does_not_warn_about_sources() -> None:
    """'Kaynaklar' başlığı yoksa source check skip — sadece akademik pattern bakılır."""
    answer = """
# Tanzimat

Tanzimat 1839'da ilan edildi.
"""
    result = validate_writer_output(answer, _ALLOWED)
    assert result.is_clean
    assert result.unknown_sources == []


def test_bold_kaynaklar_header() -> None:
    """**Kaynaklar:** kalın markdown başlığı da yakalanmalı."""
    answer = """
Bir cevap içeriği burada.

**Kaynaklar:**
- tarih_tanzimat.txt
"""
    result = validate_writer_output(answer, _ALLOWED)
    assert result.is_clean


def test_empty_answer() -> None:
    """Boş cevap → is_clean=True (uydurma yok, varsa belirsizdir)."""
    result = validate_writer_output("", _ALLOWED)
    assert result.is_clean
