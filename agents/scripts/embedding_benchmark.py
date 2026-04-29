"""Embedding model mini-benchmark — F-1 flag kararı için empirical karşılaştırma.

Amaç: SPEC.md'deki `emrecan/bert-base-turkish-cased-mean-nli-stsb-tr` (2021)
ile CONCEPT.md'nin önerdiği modern multilingual modeller arasında Türkçe
retrieval performansını karşılaştır. Sonuca göre rag/embeddings.py'da
hangi modelin kullanılacağı kararlaştırılır.

Çalıştırma:
    source .venv-agents/bin/activate
    python agents/scripts/embedding_benchmark.py

Çıktı: tabular skor (top-1, top-3, latency, vector_dim) + öneri.

Tasarım notu (Sapma defteri):
- Tek seferlik araştırma scripti → agents/scripts/ izolasyonu
- Model'ler HF cache'inde kalır; seçilen model sonra rag/embeddings.py
  içinden cache hit ile yüklenir, ek indirme yok.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# Korpus dosyası — agents/data/seed_corpus.txt; ##P<id> başlık formatı.
CORPUS_PATH = Path(__file__).resolve().parents[1] / "data" / "seed_corpus.txt"

# Karşılaştırılan modeller. SPEC önerisi (emrecan) baseline; CONCEPT'in
# önerdiği üç modern multilingual model challenger.
MODELS: list[str] = [
    "BAAI/bge-m3",
    "intfloat/multilingual-e5-large",
    "Alibaba-NLP/gte-multilingual-base",
    "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
]

# Test soruları + ground-truth paragraf ID'si.
# Sorular paraphrase yapılı: anahtar kelimeleri direkt tekrarlamayan
# semantic match testleri. Ground truth paragraf yazımı sırasında belirlendi.
QUERIES: list[tuple[str, int]] = [
    # (soru, doğru paragraf ID)
    ("Tanzimat Fermanı'nı kim ilan etti?", 0),
    ("Hareketsiz bir cisim neden hareketsiz kalır?", 3),
    ("İstiklal Marşı'nın yazarı kimdir?", 6),
    ("Eylem ile tepki arasındaki ilişki hangi yasayla anlatılır?", 5),
    ("Servet-i Fünun edebiyat akımı hangi yıllarda aktifti?", 8),
]


def load_corpus(path: Path) -> tuple[list[str], list[int]]:
    """Korpus dosyasını parse et — ##P<id> başlığını ID, sonraki paragrafı metin.

    Returns:
        (paragraf_metinleri, paragraf_id'leri) — sırayla iki list.
    """
    text = path.read_text(encoding="utf-8")
    # Satır satır gez, ##P başlığı geldikçe yeni paragraf başlat.
    paragraphs: list[str] = []
    ids: list[int] = []
    current_id: int | None = None
    current_buf: list[str] = []

    def _flush() -> None:
        # Mevcut buffer'ı paragrafa kaydet (boşsa atla).
        if current_id is not None and current_buf:
            content = " ".join(line.strip() for line in current_buf if line.strip())
            if content:
                paragraphs.append(content)
                ids.append(current_id)

    for line in text.splitlines():
        match = re.match(r"^##P(\d+)\s*$", line)
        if match:
            _flush()
            current_id = int(match.group(1))
            current_buf = []
        elif line.startswith("#"):
            # Yorum satırı (header); atla.
            continue
        else:
            current_buf.append(line)
    _flush()
    return paragraphs, ids


def evaluate_model(
    model_id: str,
    corpus: list[str],
    corpus_ids: list[int],
    queries: list[tuple[str, int]],
) -> dict[str, float | int]:
    """Bir modeli yükle, korpus + sorguları encode et, top-1/top-3 hesapla.

    Returns:
        Skor sözlüğü: model_id, vector_dim, top1, top3, latency_sec.
    """
    print(f"\n[{model_id}] yükleniyor...", flush=True)
    t0 = time.perf_counter()
    # trust_remote_code: gte-multilingual-base bunu istiyor (custom pooling).
    # bge-m3 + multilingual-e5-large + emrecan native ST uyumlu, gerek yok
    # ama tek bayrakla hepsinde güvenli.
    model = SentenceTransformer(model_id, trust_remote_code=True)
    load_time = time.perf_counter() - t0
    # ST 5.x: get_sentence_embedding_dimension → get_embedding_dimension rename.
    # Geriye uyum için fallback (eski 4.x kurulumlarında çalışsın diye).
    vector_dim = (
        model.get_embedding_dimension()
        if hasattr(model, "get_embedding_dimension")
        else model.get_sentence_embedding_dimension()
    )
    print(f"  ✓ Yüklendi ({load_time:.1f}sn, dim={vector_dim})", flush=True)

    # Encode — normalize_embeddings=True: cosine sim için L2-normalize edilmiş
    # vektörler döner; benzer hesabı saf dot product yapılabilir.
    t0 = time.perf_counter()
    corpus_emb = model.encode(
        corpus,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    query_texts = [q for q, _ in queries]
    query_emb = model.encode(
        query_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    encode_time = time.perf_counter() - t0

    # Cosine similarity — normalized vektörlerde dot product yeterli.
    # query_emb: (Q, D), corpus_emb: (C, D) → sim_matrix (Q, C).
    sim_matrix = query_emb @ corpus_emb.T

    # Her sorgu için top-3 paragraf ID'sini al.
    top1_correct = 0
    top3_correct = 0
    for qi, (_, gold_id) in enumerate(queries):
        # argsort artan; -sim_matrix ile büyükten küçüğe sırala.
        ranked = np.argsort(-sim_matrix[qi])
        # ranked indeksleri korpus pozisyonu; gerçek paragraf ID'si corpus_ids'te.
        ranked_pids = [corpus_ids[i] for i in ranked]
        if ranked_pids[0] == gold_id:
            top1_correct += 1
        if gold_id in ranked_pids[:3]:
            top3_correct += 1

    n = len(queries)
    return {
        "model_id": model_id,
        "vector_dim": vector_dim,
        "top1_acc": top1_correct / n,
        "top3_recall": top3_correct / n,
        "encode_latency_sec": encode_time,
        "load_latency_sec": load_time,
    }


def print_results_table(results: list[dict]) -> None:
    """Sonuçları okunabilir tablo halinde yazdır."""
    print("\n" + "=" * 88)
    print(f"{'Model':<55} {'dim':>5} {'top1':>6} {'top3':>6} {'enc(s)':>7}")
    print("-" * 88)
    for r in results:
        # Model adı uzunsa kısalt, görsel hizalamaya yardım.
        name = r["model_id"]
        if len(name) > 54:
            name = name[:51] + "..."
        print(
            f"{name:<55} "
            f"{r['vector_dim']:>5} "
            f"{r['top1_acc']:>5.0%} "
            f"{r['top3_recall']:>5.0%} "
            f"{r['encode_latency_sec']:>7.2f}"
        )
    print("=" * 88)


def recommend(results: list[dict]) -> str:
    """Skorlara göre öneri üret. Tie-break: önce top-1 acc, sonra top-3,
    sonra düşük dim (Qdrant collection ucuzu), sonra düşük encode latency."""
    # Sıralama anahtarı: (-top1, -top3, dim, encode_latency)
    ranked = sorted(
        results,
        key=lambda r: (
            -r["top1_acc"],
            -r["top3_recall"],
            r["vector_dim"],
            r["encode_latency_sec"],
        ),
    )
    best = ranked[0]
    return (
        f"Öneri: {best['model_id']}\n"
        f"  Top-1: {best['top1_acc']:.0%}  Top-3: {best['top3_recall']:.0%}  "
        f"Dim: {best['vector_dim']}  Encode: {best['encode_latency_sec']:.2f}sn\n"
        f"  Tie-break sırası: top-1 acc → top-3 → dim küçük → encode hızlı"
    )


def main() -> int:
    print("F-1 Embedding Mini-Benchmark — Türkçe Retrieval")
    print(f"Korpus: {CORPUS_PATH}")
    paragraphs, pids = load_corpus(CORPUS_PATH)
    print(f"  {len(paragraphs)} paragraf parse edildi (ID'ler: {pids})")
    print(f"  {len(QUERIES)} sorgu, ground-truth ID'ler: {[g for _, g in QUERIES]}")

    results: list[dict] = []
    for model_id in MODELS:
        try:
            result = evaluate_model(model_id, paragraphs, pids, QUERIES)
            results.append(result)
        except Exception as exc:
            # Bir model çökerse benchmark'ı tamamen durdurma — diğerlerini denemeye devam.
            print(f"  ✗ HATA [{model_id}]: {exc}", flush=True)
            results.append(
                {
                    "model_id": model_id,
                    "vector_dim": 0,
                    "top1_acc": 0.0,
                    "top3_recall": 0.0,
                    "encode_latency_sec": float("inf"),
                    "load_latency_sec": float("inf"),
                }
            )

    print_results_table(results)
    print()
    # Sadece başarılı çalışan model'ler arasından öner.
    valid = [r for r in results if r["vector_dim"] > 0]
    if valid:
        print(recommend(valid))
    else:
        print("Hiçbir model başarıyla çalışmadı; bağımlılık/cache durumunu kontrol et.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
