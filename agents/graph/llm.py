"""LLM backend abstraction — config-driven swap (Sapma 16).

ENV `LLM_BACKEND` üç değer alır:
- `anthropic` (dev default): claude-haiku-4-5 hızlı + ucuz, Task 3-4-5 boyunca.
- `qwen3-local`: P2 LoRA adapter (Qwen3-4B-Instruct-2507) — Linux/CUDA only.
  macOS dev'de **stub** (Sapma 17): NotImplementedError. Task 5'te etkin.
- `vllm`: vLLM serving endpoint (production gateway). Task 5+'da etkin.

Tasarım: `LLMBackend` Protocol + 3 concrete sınıf + `get_llm()` factory.
nodes.py içinden `await get_llm().generate(question, context)` ile çağrılır.

Sistem prompt'u (`_SYSTEM_PROMPT`) Türkçe pedagojik ton + grounding kuralı
içerir; tüm backend'ler aynı prompt'u kullanır → backend swap'inde tutarlılık.
"""

from __future__ import annotations

import os
from typing import Protocol

import structlog

logger = structlog.get_logger(__name__)

# Sistem prompt'u — RAG grounding + pedagojik ton + Türkçe.
# "Bağlamda olmayan bilgi uydurma" P2'de gözlenen hallucination'ı azaltır
# (P2 dersi: bilgi doğruluğu RAG context'inden gelmeli, modelden değil).
_SYSTEM_PROMPT = (
    "Sen Türkçe lise eğitim asistanısın. Görevin öğrenci seviyesine uygun, "
    "açıklayıcı ve doğru cevaplar üretmek.\n\n"
    "Kurallar:\n"
    "1. Aşağıdaki BAĞLAM bölümünden yararlanarak cevap ver.\n"
    "2. Bağlamda olmayan bilgi uydurma; emin değilsen 'yeterli bilgi yok' de.\n"
    "3. Cevabı markdown ile yapılandır (kalın başlıklar, listeler).\n"
    "4. Mümkün olduğunda bağlamdaki numaralı [1], [2] kaynaklara atıfta bulun."
)


def _build_user_message(question: str, context: str) -> str:
    """RAG context + soruyu Anthropic/vLLM mesaj formatına çevir."""
    if context:
        return f"BAĞLAM:\n{context}\n\nSORU: {question}"
    # Bağlam boşsa (retriever 0 sonuç) — modelin halüsine etme riski yüksek.
    # Validator zaten retry tetikleyecek; yine de model'e bilgi sınırı söyleyelim.
    return (
        f"SORU: {question}\n\n"
        "(Not: İlgili bağlam bulunamadı. Yeterli bilgi yoksa söyle.)"
    )


class LLMBackend(Protocol):
    """Tüm backend'lerin uyduğu protokol. nodes.py bu tipe karşı yazılır."""

    async def generate(self, question: str, context: str, max_tokens: int = 512) -> str:
        """Bağlam-grounded cevap üret. Asenkron — node çağrısı bloke etmez."""
        ...


class AnthropicBackend:
    """Claude Haiku 4.5 — dev döngüsünün primary LLM'i."""

    def __init__(
        self,
        model_id: str | None = None,
        api_key: str | None = None,
    ) -> None:
        # AsyncAnthropic — async messages.create için.
        # Import burada (lazy) — anthropic SDK ağır değil ama backend'i
        # kullanmayan run'larda import etmek gereksiz.
        from anthropic import AsyncAnthropic

        self.model_id = model_id or os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
        # api_key None → SDK ENV'den (`ANTHROPIC_API_KEY`) okur.
        self.client = AsyncAnthropic(api_key=api_key)
        logger.debug("anthropic_backend_ready", model=self.model_id)

    async def generate(self, question: str, context: str, max_tokens: int = 512) -> str:
        """Anthropic messages API ile Türkçe pedagojik cevap üret."""
        user_msg = _build_user_message(question, context)

        # `system` parametresi top-level (messages içinde değil) — Anthropic API.
        response = await self.client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        # response.content blok listesi; ilk text bloğunu al.
        # (Tool use / extended thinking olmadığı için tek text bloğu beklenir.)
        return response.content[0].text


class Qwen3LocalBackend:
    """P2 fine-tuned LoRA adapter — Qwen3-4B + bitsandbytes 4-bit.

    **Stub (Sapma 17):** macOS'ta bitsandbytes çalışmaz. Task 5'te
    Linux/CUDA path'inde implement edilecek. Yükleme kodu için referans:
    `docs/p2/P3_HANDOFF.md` § 3 (PeftModel.from_pretrained tam örnek).
    """

    def __init__(self) -> None:
        raise NotImplementedError(
            "Qwen3 lokal backend macOS'ta etkin değil (bitsandbytes "
            "Linux/CUDA-only). Dev için LLM_BACKEND=anthropic kullan; "
            "Linux/Colab T4'te Task 5+'da etkinleşecek. Detay: "
            "docs/p2/P3_HANDOFF.md § 3."
        )

    async def generate(self, question: str, context: str, max_tokens: int = 512) -> str:
        raise NotImplementedError


class VLLMBackend:
    """vLLM serving endpoint — production gateway path.

    **Stub (Sapma 17):** vLLM serving Linux GPU'da çalışır; HTTP üzerinden
    P3 client tarafa erişir. CONCEPT.md "production gateway" task'ında
    implement edilecek.
    """

    def __init__(self) -> None:
        raise NotImplementedError(
            "vLLM backend henüz etkin değil; production gateway task'ında "
            "(P3 sonu) eklenecek. Şu an LLM_BACKEND=anthropic kullan."
        )

    async def generate(self, question: str, context: str, max_tokens: int = 512) -> str:
        raise NotImplementedError


def get_llm() -> LLMBackend:
    """ENV `LLM_BACKEND`'e göre uygun backend'i instantiate et.

    Returns:
        LLMBackend protokolüne uyan concrete instance.

    Raises:
        ValueError: Bilinmeyen backend adı.
        NotImplementedError: qwen3-local veya vllm seçilirse (henüz stub).
    """
    backend = os.getenv("LLM_BACKEND", "anthropic").lower()
    if backend == "anthropic":
        return AnthropicBackend()
    if backend == "qwen3-local":
        return Qwen3LocalBackend()
    if backend == "vllm":
        return VLLMBackend()
    raise ValueError(
        f"Bilinmeyen LLM_BACKEND: '{backend}'. "
        "Geçerli değerler: anthropic | qwen3-local | vllm"
    )
