"""Crew smoke test — multi-disciplinary soru ile Researcher + Writer akışı.

TASKS.md test sorusu: Newton hareket yasaları + Osmanlı modernleşmesi
arasında benzer dinamik. İki farklı disiplin → CrewAI mimarisinin
gerçek değer alanı.

Kullanım:
    docker-compose up qdrant -d
    source .venv-agents/bin/activate
    PYTHONPATH=. python -m agents.crew.test_crew

Önkoşul: agents/scripts/index_seed.py çalıştırılmış olmalı
(tarih_tanzimat.txt + fizik_newton.txt indeksli).
"""

from __future__ import annotations

import os
from pathlib import Path

from crewai import Crew, Process
from dotenv import load_dotenv

from agents.crew.agents import create_researcher_agent, create_writer_agent
from agents.crew.tasks import create_research_task, create_writing_task
from agents.crew.tools import search_education_materials
from agents.crew.validators import validate_writer_output


def _load_env_cascade() -> None:
    """pipeline.py ile aynı pattern: agents/.env > root/.env > ml/.env."""
    repo_root = Path(__file__).resolve().parents[2]
    for path in [
        repo_root / "agents" / ".env",
        repo_root / ".env",
        repo_root / "ml" / ".env",
    ]:
        if path.exists():
            load_dotenv(path)


def main() -> int:
    _load_env_cascade()
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("✗ ANTHROPIC_API_KEY yok. agents/.env veya ml/.env'e ekle.")
        return 1

    question = (
        "Newton'un hareket yasaları ve Osmanlı'nın modernleşme süreci arasında "
        "benzer bir dinamik var mı? Açıkla."
    )

    print("=" * 80)
    print(f"Soru: {question}")
    print("=" * 80)
    print()

    # Agent'ları oluştur — Researcher RAG tool'lu, Writer tool'suz.
    researcher = create_researcher_agent(search_education_materials)
    writer = create_writer_agent()

    # Task'ları oluştur — Writer task Researcher'ın çıktısına bağlı.
    research_task = create_research_task(researcher, question)
    writing_task = create_writing_task(writer, research_task)

    # Crew kickoff — sequential process: önce researcher, sonra writer.
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()

    print("\n" + "=" * 80)
    print("FINAL CEVAP (Writer çıktısı):")
    print("=" * 80)
    # CrewOutput.raw / .pydantic / .json_dict erişilebilir; raw text default.
    answer_text = result.raw if hasattr(result, "raw") else str(result)
    print(answer_text)

    print("\n" + "=" * 80)
    print(
        f"Token kullanımı: {result.token_usage if hasattr(result, 'token_usage') else 'N/A'}"
    )

    # Sapma 37 — Writer çıktısını post-validate et (Sapma 24 follow-up).
    # Allowed sources: index_seed.py'de yüklü dosyalar (bilinen whitelist).
    # Production'da Researcher tool çağrılarından dinamik toplanır.
    print("\n" + "=" * 80)
    print("Post-validator (Sapma 24 follow-up):")
    validation = validate_writer_output(
        answer_text=answer_text,
        allowed_sources={"tarih_tanzimat.txt", "fizik_newton.txt"},
    )
    if validation.is_clean:
        print("  ✓ Cevap temiz — uydurma kaynak tespit edilmedi.")
    else:
        print(f"  ⚠ {len(validation.warnings)} uyarı:")
        for w in validation.warnings:
            print(f"    {w}")

    print("\n✓ Crew smoke test tamamlandı")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
