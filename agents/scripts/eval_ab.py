"""Eval set A/B test — RAG-with vs baseline Anthropic karşılaştırması.

P2'nin `ml/data/processed/eval.jsonl`'inden N random örnek alır:
- **A (baseline)**: claude-haiku-4-5 doğrudan (sistem prompt + soru, RAG yok)
- **B (RAG)**: LangGraph pipeline (train.jsonl → Qdrant retrieve → generate)

train.jsonl'in **output**'ları Qdrant'a chunk olarak indekslenir → RAG için
gerçek corpus. eval.jsonl ayrı (eval-train leak yok).

Metrikler P2 ile aynı:
- ROUGE-1, ROUGE-L (kelime overlap)
- BERTScore F1 (Türkçe semantic similarity, `bert-base-multilingual-cased`)
- Latency (sec)
- Token count (Anthropic API response usage)

Çıktı:
- `agents/data/eval_ab_results.csv` — sample bazlı + manuel rating slot
- Console summary: ortalama metrikler, RAG win rate, latency

Kullanım:
    docker-compose up qdrant -d
    source .venv-agents/bin/activate
    PYTHONPATH=. python agents/scripts/eval_ab.py [--n 30] [--seed 42]
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import structlog
from anthropic import AsyncAnthropic
from bert_score import score as bert_score_compute
from dotenv import load_dotenv
from rouge_score import rouge_scorer

from agents.graph.llm import _SYSTEM_PROMPT
from agents.graph.pipeline import build_pipeline
from agents.rag.embeddings import TurkishEmbedder
from agents.rag.indexer import DocumentIndexer

logger = structlog.get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_PATH = REPO_ROOT / "ml" / "data" / "processed" / "eval.jsonl"
TRAIN_PATH = REPO_ROOT / "ml" / "data" / "processed" / "train.jsonl"
EVAL_COLLECTION = "eduai_eval_corpus"  # eval-spesifik collection (production'dan ayrı)
RESULTS_CSV = REPO_ROOT / "agents" / "data" / "eval_ab_results.csv"


def _load_env() -> None:
    """`.env` cascade — pipeline.py ile aynı pattern."""
    for path in [
        REPO_ROOT / "agents" / ".env",
        REPO_ROOT / ".env",
        REPO_ROOT / "ml" / ".env",
    ]:
        if path.exists():
            load_dotenv(path)
            break


def load_jsonl(path: Path) -> list[dict]:
    """Standard JSONL loader."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def index_train_corpus(samples: list[dict]) -> DocumentIndexer:
    """train.jsonl outputs'unu Qdrant'a `eduai_eval_corpus` collection'ına yükle.

    Her train sample bir chunk olarak indekslenir; instruction + output
    birleştirilir (LLM context'inde "Soru: X. Cevap: Y." formatında zengin
    bilgi). Subject metadata korunur — RAG retriever subject filter
    eval'de eşleşsin.
    """
    embedder = TurkishEmbedder()
    indexer = DocumentIndexer(
        collection_name=EVAL_COLLECTION,
        embedder=embedder,
    )
    print(
        f"  📚 Train corpus indeksleniyor: {len(samples)} sample → '{EVAL_COLLECTION}'"
    )

    # Tek tek index_file çağırmak verimsiz (her biri için Qdrant upsert).
    # Doğrudan internal API ile batch encode + upsert.
    # Ama public API'yi koru: her sample için tempfile + index_file mantıksız.
    # Pragmatik: collection'a doğrudan upsert (indexer yardımcılarını kullanmadan).

    # Deterministik doc_id'ler için stem = "train_<idx>".
    # Her sample tek chunk (kısa Q&A); splitter çağırmıyoruz.
    import uuid

    from qdrant_client.http import models as qmodels

    from agents.rag.indexer import _POINT_NAMESPACE

    texts = [f"Soru: {s['instruction']}\n\nCevap: {s['output']}" for s in samples]

    # Toplu encode (büyük batch'te ms_per_doc düşer).
    print(f"  ⚙️  {len(texts)} chunk embed ediliyor (e5-large)...")
    t0 = time.perf_counter()
    vectors = embedder.embed_documents(texts)
    encode_time = time.perf_counter() - t0
    print(
        f"  ✓ Embed bitti ({encode_time:.1f}sn, {encode_time / len(texts) * 1000:.0f}ms/sample)"
    )

    points = []
    for i, (sample, text, vector) in enumerate(zip(samples, texts, vectors)):
        doc_id = f"train_{i}"
        payload = {
            "source": f"train.jsonl#{i}",
            "page_num": 0,
            "subject": sample.get("subject", "genel"),
            "grade_level": sample.get("grade", 0),
            "chunk_index": 0,
            "doc_id": doc_id,
            "text": text,
        }
        point_id = str(uuid.uuid5(_POINT_NAMESPACE, f"{doc_id}:0"))
        points.append(qmodels.PointStruct(id=point_id, vector=vector, payload=payload))

    # Qdrant idempotent upsert — eval re-run'larında re-index gereksiz.
    indexer.client.upsert(collection_name=EVAL_COLLECTION, points=points, wait=True)
    print(f"  ✓ {len(points)} chunk Qdrant'a yüklendi\n")
    return indexer


async def baseline_anthropic(
    client: AsyncAnthropic,
    model: str,
    question: str,
) -> tuple[str, dict]:
    """A (baseline): RAG yok, doğrudan claude-haiku-4-5.

    Aynı sistem prompt LangGraph ile (Türkçe pedagojik ton); fark sadece
    BAĞLAM: bölümünün yokluğu. Bu temiz A/B karşılaştırması: "RAG context
    eklemek kaliteyi ne kadar artırıyor?"

    Returns:
        (answer_text, usage_dict) — usage_dict input/output token'ları içerir.
    """
    response = await client.messages.create(
        model=model,
        max_tokens=512,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"SORU: {question}"}],
    )
    answer = response.content[0].text
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return answer, usage


async def rag_pipeline_run(pipeline, question: str, subject: str, grade: int) -> dict:
    """B (RAG): LangGraph pipeline.

    Pipeline EVAL_COLLECTION kullanmıyor — default `eduai_documents`.
    Eval için collection swap'ı: ENV `QDRANT_COLLECTION=eduai_eval_corpus`
    set edip retriever singleton'ı yenile. Çağıran zaten yapmış olur.
    """
    state = {
        "question": question,
        "subject": subject,
        "grade_level": grade,
        "session_id": "eval-ab",
        "attempts": 0,
        "needs_retry": False,
    }
    return await pipeline.ainvoke(state)


def compute_metrics(prediction: str, reference: str) -> dict[str, float]:
    """ROUGE-1 + ROUGE-L + BERTScore F1 (lazy bert call dışarıdan, batch'te)."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    scores = scorer.score(reference, prediction)
    return {
        "rouge1_f": scores["rouge1"].fmeasure,
        "rougeL_f": scores["rougeL"].fmeasure,
    }


def compute_bertscore_batch(
    predictions: list[str], references: list[str]
) -> list[float]:
    """BERTScore F1 batch — ek model load tek seferlik, multi-doc verimli.

    `bert-base-multilingual-cased` Türkçe için yeterli; P2 evaluate.py
    aynı modeli kullanmıştı.
    """
    print("  ⚙️  BERTScore hesaplanıyor (multilingual-bert)...")
    P, R, F1 = bert_score_compute(  # noqa: N806
        predictions,
        references,
        lang="tr",
        verbose=False,
        batch_size=8,
    )
    return F1.tolist()


async def main_async(n: int, seed: int) -> int:
    _load_env()
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("✗ ANTHROPIC_API_KEY yok. agents/.env'a ekle.")
        return 1

    print(f"=== Eval A/B Test: {n} sample, seed={seed} ===\n")

    # 1) Eval setini yükle + sample
    eval_data = load_jsonl(EVAL_PATH)
    print(f"Eval set: {len(eval_data)} toplam örnek")
    random.seed(seed)
    samples = random.sample(eval_data, k=n)
    print(f"Random sample: {n} örnek seçildi (seed={seed})\n")

    # 2) Train corpus'u Qdrant'a indeksle (eval-spesifik collection)
    train_data = load_jsonl(TRAIN_PATH)
    index_train_corpus(train_data)

    # 3) Pipeline'ı eval collection'ına yönlendir — singleton reset
    os.environ["QDRANT_COLLECTION"] = EVAL_COLLECTION
    import agents.graph.nodes as nodes_module

    nodes_module._retriever_singleton = None  # autouse fixture yok burada
    pipeline = build_pipeline()

    # Anthropic client baseline için (paylaşılan singleton)
    anthropic_client = AsyncAnthropic()
    anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")

    # 4) Her sample için A + B çalıştır
    results: list[dict[str, Any]] = []
    print(f"--- A/B run ({n} sample × 2 kondisyon = {n * 2} LLM call) ---\n")

    for i, sample in enumerate(samples, start=1):
        question = sample["instruction"]
        gold = sample["output"]
        subject = sample.get("subject", "genel")
        grade = sample.get("grade", 9)

        print(f"[{i}/{n}] ({subject}) {question[:60]}...")

        # A — baseline
        t0 = time.perf_counter()
        try:
            a_answer, a_usage = await baseline_anthropic(
                anthropic_client, anthropic_model, question
            )
            a_latency = time.perf_counter() - t0
            a_error = ""
        except Exception as exc:
            a_answer = ""
            a_usage = {"input_tokens": 0, "output_tokens": 0}
            a_latency = time.perf_counter() - t0
            a_error = str(exc)

        # B — RAG
        t0 = time.perf_counter()
        try:
            b_result = await rag_pipeline_run(pipeline, question, subject, grade)
            b_answer = b_result.get("answer", "")
            b_confidence = float(b_result.get("confidence", 0.0))
            b_attempts = int(b_result.get("attempts", 0))
            b_latency = time.perf_counter() - t0
            b_error = ""
        except Exception as exc:
            b_answer = ""
            b_confidence = 0.0
            b_attempts = 0
            b_latency = time.perf_counter() - t0
            b_error = str(exc)

        # ROUGE
        a_rouge = (
            compute_metrics(a_answer, gold)
            if a_answer
            else {"rouge1_f": 0.0, "rougeL_f": 0.0}
        )
        b_rouge = (
            compute_metrics(b_answer, gold)
            if b_answer
            else {"rouge1_f": 0.0, "rougeL_f": 0.0}
        )

        results.append(
            {
                "idx": i,
                "subject": subject,
                "grade": grade,
                "question": question,
                "gold": gold,
                "a_answer": a_answer,
                "a_latency": round(a_latency, 2),
                "a_input_tokens": a_usage["input_tokens"],
                "a_output_tokens": a_usage["output_tokens"],
                "a_rouge1": round(a_rouge["rouge1_f"], 4),
                "a_rougeL": round(a_rouge["rougeL_f"], 4),
                "a_error": a_error,
                "b_answer": b_answer,
                "b_latency": round(b_latency, 2),
                "b_confidence": round(b_confidence, 4),
                "b_attempts": b_attempts,
                "b_rouge1": round(b_rouge["rouge1_f"], 4),
                "b_rougeL": round(b_rouge["rougeL_f"], 4),
                "b_error": b_error,
                "a_rating_manual": "",  # kullanıcı 1-5 dolduracak
                "b_rating_manual": "",
            }
        )

    # 5) BERTScore batch (ek model load — tek seferlik)
    print("\n--- BERTScore hesaplama (batch) ---")
    a_preds = [r["a_answer"] for r in results]
    b_preds = [r["b_answer"] for r in results]
    refs = [r["gold"] for r in results]
    a_bert = compute_bertscore_batch(a_preds, refs)
    b_bert = compute_bertscore_batch(b_preds, refs)
    for r, ab, bb in zip(results, a_bert, b_bert):
        r["a_bertscore"] = round(ab, 4)
        r["b_bertscore"] = round(bb, 4)

    # 6) CSV yaz
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())
    with RESULTS_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  ✓ Sample-bazlı sonuçlar: {RESULTS_CSV}")

    # 7) Özet tablosu
    def avg(key: str) -> float:
        vals = [r[key] for r in results if isinstance(r[key], int | float)]
        return sum(vals) / len(vals) if vals else 0.0

    a_total_tokens = sum(r["a_input_tokens"] + r["a_output_tokens"] for r in results)
    b_wins_rouge1 = sum(1 for r in results if r["b_rouge1"] > r["a_rouge1"])
    b_wins_rougeL = sum(1 for r in results if r["b_rougeL"] > r["a_rougeL"])
    b_wins_bert = sum(1 for r in results if r["b_bertscore"] > r["a_bertscore"])

    print("\n" + "=" * 80)
    print(f"ÖZET — {n} sample (seed={seed})")
    print("=" * 80)
    print(f"{'Metric':<20} {'A (baseline)':>15} {'B (RAG)':>15} {'Δ (B-A)':>12}")
    print("-" * 80)
    for label, ka, kb in [
        ("ROUGE-1 F", "a_rouge1", "b_rouge1"),
        ("ROUGE-L F", "a_rougeL", "b_rougeL"),
        ("BERTScore F1", "a_bertscore", "b_bertscore"),
        ("Latency (sn)", "a_latency", "b_latency"),
    ]:
        a_avg, b_avg = avg(ka), avg(kb)
        delta = b_avg - a_avg
        sign = "+" if delta >= 0 else ""
        print(f"{label:<20} {a_avg:>15.4f} {b_avg:>15.4f} {sign}{delta:>11.4f}")
    print("-" * 80)
    print(f"{'B win-rate (ROUGE-1)':<20} {b_wins_rouge1}/{n} ({b_wins_rouge1 / n:.0%})")
    print(f"{'B win-rate (ROUGE-L)':<20} {b_wins_rougeL}/{n} ({b_wins_rougeL / n:.0%})")
    print(f"{'B win-rate (BERTScore)':<20} {b_wins_bert}/{n} ({b_wins_bert / n:.0%})")
    print(f"{'A toplam token':<20} {a_total_tokens}")
    print(f"{'B avg confidence':<20} {avg('b_confidence'):.4f}")
    print(f"{'B avg attempts':<20} {avg('b_attempts'):.2f}")
    print("=" * 80)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30, help="Sample size (default 30)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    args = parser.parse_args()
    return asyncio.run(main_async(args.n, args.seed))


if __name__ == "__main__":
    raise SystemExit(main())
