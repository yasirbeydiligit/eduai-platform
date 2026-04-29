"""Embedding model HARD benchmark — F-1 kararı için ayrım gücü testi.

İlk benchmark (embedding_benchmark.py) %100 tie verdi: test data semantic
ayrımı zorlamıyordu. Bu sürüm:
  - 21 paragraf (vs 9), her konuda DISTRACTOR (yakın ama yanlış)
  - 10 paraphrase-ağır soru: anahtar kelime tekrarı az, anlamsal eşleme zor
  - Sadece 2 finalist: intfloat/multilingual-e5-large vs emrecan SPEC baseline
  - bge-m3 latency 65sn'den elendi; gte custom modeling.py uyumsuzluğundan elendi

Çalıştırma:
    source .venv-agents/bin/activate
    python agents/scripts/embedding_benchmark_hard.py
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

CORPUS_PATH = Path(__file__).resolve().parents[1] / "data" / "seed_corpus_hard.txt"

# Finaller — first benchmark sonuçlarına göre filtrelendi.
MODELS: list[str] = [
    "intfloat/multilingual-e5-large",
    "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
]

# E5 ailesi instruction prefix gerektirir: query/passage ayrımı.
# Diğer modeller için boş prefix; doğru karşılaştırma için her model kendi
# prefix kuralını uygular.
PREFIXES: dict[str, dict[str, str]] = {
    "intfloat/multilingual-e5-large": {"query": "query: ", "passage": "passage: "},
    # emrecan saf BERT-Turkish; prefix yok.
    "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr": {"query": "", "passage": ""},
}

# 10 zor soru. Her biri paraphrase ağır + en az 1 yakın distractor karşıt.
# Format: (soru, doğru_paragraf_id, kısa_açıklama)
QUERIES: list[tuple[str, int, str]] = [
    (
        "Otomobil aniden frenlediğinde içindeki yolcuların ileri savrulması hangi temel fizik kuralının doğal sonucudur?",
        6,
        "Newton I (eylemsizlik); distractor P7/P8/P10",
    ),
    (
        "Bir uzay aracının itki sistemi hangi klasik mekanik yasasını teknolojiye uyarlayarak çalışır?",
        8,
        "Newton III (etki-tepki); distractor P7 (F=ma)",
    ),
    (
        "Padişah otoritesinin yazılı bir metinle ilk kez kısıtlandığı ve halka değil yerel güç odaklarına yönelik 1808 belgesi hangisidir?",
        0,
        "Sened-i İttifak; distractor P3 (Kanun-ı Esasi yine yetki sınırlıyor)",
    ),
    (
        "Osmanlı'da gayrimüslim tebaaya memurluk, askerlik ve adli eşitlik tanıyan, Avrupa baskısı altında hazırlanmış belge hangisidir?",
        2,
        "Islahat Fermanı; distractor P1 (Tanzimat aynı padişah)",
    ),
    (
        "Anayasanın yeniden işler hale gelmesi ve parlamentonun 1908'de yeniden açılmasıyla başlayan dönem nasıl adlandırılır?",
        4,
        "II. Meşrutiyet; distractor P3 (I. Meşrutiyet de Kanun-ı Esasi)",
    ),
    (
        "Sürtünmesiz ortamda hareketin sonsuza dek süreceği fikrini deneylerle ortaya koyan ve Newton'a zemin hazırlayan İtalyan bilim insanı kimdir?",
        10,
        "Galilei; distractor P6 (Newton I bu fikrin formel halidir)",
    ),
    (
        "Hayatının son yıllarını Kahire'de geçirmiş, milli mücadele ruhunu yansıtan destansı şiiriyle tanınan şair kimdir?",
        11,
        "Mehmet Akif; distractor P12 (Yahya Kemal — diplomat ama Mısır değil)",
    ),
    (
        "Sanatın toplumsal işlevini ön plana çıkaran ve Türk edebiyatına gazete-tiyatro-roman türlerini ilk getiren akım hangisidir?",
        14,
        "Tanzimat edebiyatı; distractor P13 (Servet-i Fünun aynı dönem yakın)",
    ),
    (
        "Mai ve Siyah ile Aşk-ı Memnu romanlarının yazarı olan ve Servet-i Fünun grubunun roman alanındaki en önemli temsilcisi kimdir?",
        16,
        "Halit Ziya; distractor P13 (grup paragrafı kendisini de anıyor)",
    ),
    (
        "Bir fonksiyonun bir noktadaki anlık değişim oranını veren ve teğet doğrusunun eğimine karşılık gelen kavram nedir?",
        19,
        "Türev; distractor P20 (integral — birikim, türev'in tersi)",
    ),
]


def load_corpus(path: Path) -> tuple[list[str], list[int]]:
    """Korpus dosyasını parse — embedding_benchmark.py ile aynı format."""
    text = path.read_text(encoding="utf-8")
    paragraphs: list[str] = []
    ids: list[int] = []
    current_id: int | None = None
    current_buf: list[str] = []

    def _flush() -> None:
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
            continue
        else:
            current_buf.append(line)
    _flush()
    return paragraphs, ids


def evaluate_model(
    model_id: str,
    corpus: list[str],
    corpus_ids: list[int],
    queries: list[tuple[str, int, str]],
) -> dict:
    """Model yükle, prefix uygula, encode, top-1/top-3 hesapla + her soru
    için doğru/yanlış detayını topla (qualitative inspection için)."""
    print(f"\n[{model_id}] yükleniyor...", flush=True)
    t0 = time.perf_counter()
    model = SentenceTransformer(model_id, trust_remote_code=True)
    load_time = time.perf_counter() - t0

    vector_dim = (
        model.get_embedding_dimension()
        if hasattr(model, "get_embedding_dimension")
        else model.get_sentence_embedding_dimension()
    )
    print(f"  ✓ Yüklendi ({load_time:.1f}sn, dim={vector_dim})", flush=True)

    prefix = PREFIXES.get(model_id, {"query": "", "passage": ""})
    corpus_texts = [prefix["passage"] + p for p in corpus]
    query_texts = [prefix["query"] + q for q, _, _ in queries]

    t0 = time.perf_counter()
    corpus_emb = model.encode(
        corpus_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    query_emb = model.encode(
        query_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    encode_time = time.perf_counter() - t0

    sim_matrix = query_emb @ corpus_emb.T

    top1_correct = 0
    top3_correct = 0
    per_query: list[dict] = []
    for qi, (qtext, gold_id, note) in enumerate(queries):
        ranked = np.argsort(-sim_matrix[qi])
        ranked_pids = [corpus_ids[i] for i in ranked]
        top1_pid = ranked_pids[0]
        top3_pids = ranked_pids[:3]
        top1_hit = top1_pid == gold_id
        top3_hit = gold_id in top3_pids
        if top1_hit:
            top1_correct += 1
        if top3_hit:
            top3_correct += 1
        per_query.append(
            {
                "qi": qi,
                "gold": gold_id,
                "top1": top1_pid,
                "top3": top3_pids,
                "top1_score": float(sim_matrix[qi][ranked[0]]),
                "gold_score": float(sim_matrix[qi][corpus_ids.index(gold_id)]),
                "ok": top1_hit,
                "note": note,
            }
        )

    n = len(queries)
    return {
        "model_id": model_id,
        "vector_dim": vector_dim,
        "top1_acc": top1_correct / n,
        "top3_recall": top3_correct / n,
        "encode_latency_sec": encode_time,
        "load_latency_sec": load_time,
        "per_query": per_query,
    }


def print_qualitative(result: dict, queries: list[tuple[str, int, str]]) -> None:
    """Her soru için doğru/yanlış + skor farkı."""
    print(f"\n  Detay [{result['model_id']}]:")
    for r in result["per_query"]:
        sym = "✓" if r["ok"] else "✗"
        gap = r["top1_score"] - r["gold_score"]
        gap_str = (
            f"(top1-gold gap: {gap:+.3f})"
            if not r["ok"]
            else f"(score: {r['top1_score']:.3f})"
        )
        qtext = queries[r["qi"]][0]
        # Soruyu kısalt; konsol okumasına yardım.
        if len(qtext) > 60:
            qtext = qtext[:57] + "..."
        print(f"    {sym} Q{r['qi']}: gold=P{r['gold']} top1=P{r['top1']} {gap_str}")
        print(f"      «{qtext}»")
        print(f"      [{r['note']}]")


def main() -> int:
    print("F-1 HARD Mini-Benchmark — Türkçe Retrieval (zor distractor + paraphrase)")
    print(f"Korpus: {CORPUS_PATH}")
    paragraphs, pids = load_corpus(CORPUS_PATH)
    print(f"  {len(paragraphs)} paragraf parse edildi")
    print(f"  {len(QUERIES)} sorgu (paraphrase ağır)")

    results: list[dict] = []
    for model_id in MODELS:
        try:
            result = evaluate_model(model_id, paragraphs, pids, QUERIES)
            results.append(result)
        except Exception as exc:
            print(f"  ✗ HATA [{model_id}]: {exc}", flush=True)
            results.append(
                {
                    "model_id": model_id,
                    "vector_dim": 0,
                    "top1_acc": 0.0,
                    "top3_recall": 0.0,
                    "encode_latency_sec": float("inf"),
                    "load_latency_sec": float("inf"),
                    "per_query": [],
                }
            )

    # Özet tablo
    print("\n" + "=" * 88)
    print(f"{'Model':<55} {'dim':>5} {'top1':>6} {'top3':>6} {'enc(s)':>7}")
    print("-" * 88)
    for r in results:
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

    # Detaylı per-query (qualitative inspection)
    for r in results:
        if r["per_query"]:
            print_qualitative(r, QUERIES)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
