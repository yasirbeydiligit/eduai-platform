"""CrewAI Task tanımları — Researcher araştırma + Writer yazma.

CrewAI 1.x Task imzası: `description`, `expected_output`, `agent`, `context`
(önceki task çıktıları). Sequential process'te Writer task `context=[research]`
ile Researcher çıktısını otomatik alır.
"""

from __future__ import annotations

from crewai import Agent, Task


def create_research_task(researcher: Agent, question: str) -> Task:
    """Researcher'a verilen araştırma görevi.

    description'da `{question}` doğrudan gömülü (CrewAI input
    interpolation'ı task seviyesinde değil crew seviyesinde aktif olduğu
    için). expected_output Researcher'ın yapısını öğrenir.
    """
    return Task(
        description=(
            f"Şu soru için Türk lise müfredatından kaynak topla: "
            f'"{question}"\n\n'
            "Adımlar:\n"
            "1. Soruda hangi disiplinler/konular geçiyor onları belirle.\n"
            "2. Her disiplin için search_education_materials tool'unu çağır "
            "(uygun subject filtresiyle).\n"
            "3. Bulunan kaynakları paragraf paragraf özetle; her özet için "
            "kaynak dosya adını [kaynak: x.txt] formatında belirt."
        ),
        expected_output=(
            "Konuyla ilgili 3-6 paragraflık kaynak özeti. Her paragraf "
            "kısa (3-5 cümle) ve sonunda kaynak referansı içerir. Disiplin "
            "değiştiğinde paragraf da değişir."
        ),
        agent=researcher,
    )


def create_writing_task(writer: Agent, research_task: Task) -> Task:
    """Writer'a verilen yazma görevi — Researcher çıktısını input alır.

    Sapma 24 fix: 'Kaynaklar' bölümünde hallucination'ı önleyen explicit
    kurallar; sadece bağlamda görülen dosya adlarına izin var.
    """
    return Task(
        description=(
            "Araştırmacının topladığı kaynakları kullanarak öğrenciye "
            "yönelik açık, pedagojik bir cevap yaz.\n\n"
            "İçerik kuralları:\n"
            "1. Markdown başlıklar ve listeler kullan.\n"
            "2. Bağlamda olmayan bilgi UYDURMA; emin değilsen 'kaynaklarda "
            "yer almıyor' de.\n"
            "3. Disiplinler arası bağlantı varsa açıkça vurgula "
            "(örn. 'Tıpkı Newton birinci yasasındaki gibi, Tanzimat...').\n"
            "4. 250-450 kelime hedefle — fazla uzatma.\n\n"
            "KAYNAKLAR bölümü kuralları (KRİTİK — sıkı uy):\n"
            "5. SADECE Researcher çıktısında [kaynak: <dosya>.txt] formatında "
            "geçen dosya adlarını listele.\n"
            "6. Kitap adı, yazar, yayın tarihi, yayınevi, ISBN, dergi, link "
            "EKLEME — bunlar yoksa uydurma. Sadece dosya adı + kısa içerik "
            "etiketi yeterli (örn. '- fizik_newton.txt: Newton hareket yasaları').\n"
            "7. Akademik referans formatı (APA/MLA) KULLANMA — burada akademik "
            "yayın değil, RAG bağlamı listeleniyor."
        ),
        expected_output=(
            "Markdown formatında, lise seviyesine uygun, kaynak-temelli "
            "Türkçe cevap. Disiplinler arası analoji açıkça yapılmış. "
            "Sonunda 'Kaynaklar:' başlığı; SADECE bağlamdaki dosya adları "
            "(uydurma yayın bilgisi yok)."
        ),
        agent=writer,
        # context: researcher'ın çıktısı writer'ın prompt'una eklenir.
        # Sequential process bunu otomatik yapsa da explicit context daha temiz.
        context=[research_task],
    )
