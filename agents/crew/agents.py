"""CrewAI agent factory'leri — Researcher + Writer rolleri.

P3 mimarisinde CrewAI **karmaşık (multi-disciplinary) sorular** için aktif:
basit sorularda LangGraph pipeline yeter, ancak farklı disiplinleri köprüleyen
soruda Researcher kaynak toplar → Writer pedagojik metin yazar.

LLM: CrewAI 1.x LiteLLM wrapper kullanır. Model string'i provider prefix'li:
`anthropic/claude-haiku-4-5`. `ANTHROPIC_API_KEY` ENV otomatik çekilir.
ENV `ANTHROPIC_MODEL` ile model swap kolay.

Sapma 22 (CrewAI 1.x major bump): SPEC `crewai>=0.80.0`'di; F-2 review
sırasında 1.14.3 yüklü olduğu görüldü. 0.x agent constructor (kwargs) ile
1.x Pydantic-driven API uyumlu kaldığından minimal kod değişikliği yetti;
LLM provider config sadece `LLM(model="anthropic/...")` şekline geçti.
"""

from __future__ import annotations

import os

from crewai import LLM, Agent
from crewai.tools import BaseTool


def _build_llm() -> LLM:
    """Agent'lara verilecek LiteLLM wrapper'ı.

    LiteLLM model string formatı: `<provider>/<model>`. Provider prefix
    yoksa LiteLLM auto-detect dener; explicit yapmak production'da daha
    güvenilir.
    """
    model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
    # api_key ENV'den otomatik (ANTHROPIC_API_KEY); LiteLLM bilir.
    return LLM(model=f"anthropic/{model}")


def create_researcher_agent(rag_tool: BaseTool) -> Agent:
    """Eğitim İçerik Araştırmacısı — RAG ile kaynak toplar.

    Args:
        rag_tool: `search_education_materials` decorator-tool veya BaseTool.
                  Agent doğal dilden tool'u çağırır.
    """
    return Agent(
        role="Eğitim İçerik Araştırmacısı",
        goal=(
            "Soruyla ilgili Türk lise müfredatından doğru, güncel ve "
            "öğrenci seviyesine uygun kaynakları bul. Birden fazla disiplin "
            "kapsanıyorsa her disiplin için ayrı arama yap."
        ),
        backstory=(
            "Türk Milli Eğitim müfredatına hakim, lise düzeyinde tarih, "
            "fizik, matematik ve edebiyat içeriklerini iyi bilen bir "
            "araştırmacısın. Kaynaklara dayanmadan yorum yapmazsın; "
            "her iddianı bir dökümandan dayandırırsın."
        ),
        tools=[rag_tool],
        llm=_build_llm(),
        verbose=True,
        # max_iter: tool çağrısı + reasoning döngüsü limiti.
        # 3: araştırma + 2 retry yeterli; daha çok sonsuz loop riski.
        max_iter=3,
        allow_delegation=False,
    )


def create_writer_agent() -> Agent:
    """Pedagojik İçerik Yazarı — researcher çıktısını işlenir cevap haline getirir."""
    return Agent(
        role="Pedagojik İçerik Yazarı",
        goal=(
            "Araştırmacının topladığı kaynaklara dayanarak, lise öğrencisinin "
            "kolayca anlayacağı dilde, markdown formatında düzenli bir cevap "
            "üret. Disiplinler arası bağlantıları açıkça kur."
        ),
        backstory=(
            "10 yıllık lise öğretmenliği deneyimine sahipsin. Karmaşık "
            "kavramları somut örneklerle açıklamayı, farklı dersler arasında "
            "köprü kurmayı seversin. Bağlamda olmayan iddialar yapmaz, "
            "kaynak göstermeden çıkarım yazmazsın. UYDURMA YAYIN BİLGİSİ "
            "(kitap adı, yazar, tarih, yayınevi) yazmazsın — sadece sana "
            "gerçekten verilen dosya adlarını referans gösterirsin."
        ),
        # Writer tool kullanmaz; sadece Researcher'ın context'ini işler.
        llm=_build_llm(),
        verbose=True,
        max_iter=2,
        allow_delegation=False,
    )
