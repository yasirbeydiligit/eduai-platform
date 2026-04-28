"""
data_prep.py — EduAI P2 Türkçe lise Q&A veri üretici.

Strateji: Seçenek A (Sentetik, Claude API).
Türkiye MEB güncel öğretim programlarına uygun soru-cevap çiftleri üretir,
kalite filtrelerinden geçirir ve stratified %80/%20 split ile JSONL dosyalarına yazar.

Kullanım:
    python training/data_prep.py                    # default: 50 örnek/subject, Haiku
    python training/data_prep.py --target 30        # subject başına 30 örnek
    python training/data_prep.py --model sonnet     # yüksek kalite, ~4x maliyet
    python training/data_prep.py --semantic-dedup   # ek semantic deduplication
    python training/data_prep.py --dry-run          # API call'suz mock pipeline testi

Çıktılar:
    data/raw/eduai_qa_raw.jsonl         — Claude'dan gelen ham response'lar (debug)
    data/processed/train.jsonl          — eğitim seti (%80)
    data/processed/eval.jsonl           — değerlendirme seti (%20)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

import anthropic
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

# --- Sabit yapılandırma --------------------------------------------------

# Reproducibility — MLflow karşılaştırmasında gürültüyü azaltır.
SEED = 42

# Proje kökü `ml/` — path'leri ona göre çözüyoruz.
ML_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ML_ROOT / "data" / "raw" / "eduai_qa_raw.jsonl"
TRAIN_PATH = ML_ROOT / "data" / "processed" / "train.jsonl"
EVAL_PATH = ML_ROOT / "data" / "processed" / "eval.jsonl"

# Claude model alias'ları — hız/maliyet/kalite dengesi:
#   haiku:  ucuz + hızlı + Türkçe yeterli (default; ~450 örnek ≈ $1)
#   sonnet: orta maliyet, daha nüanslı pedagojik ton (~$4)
#   opus:   premium; sadece son cila / QA aşamalarında mantıklı
MODEL_ALIASES = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-7",
}
DEFAULT_MODEL = "haiku"

# --- MEB müfredatı konu seed'leri ---------------------------------------
# NOT: Spesifik kazanım kodları (örn. "9.1.1.1") prompt'a eklenmiyor; MEB
# müfredatı periyodik revize ediliyor ve kod formatı dönem bazında değişiyor.
# Bunun yerine konu-seviye seed veriyoruz — Claude bu çerçeveden expand eder.
# Prompt'ta "MEB güncel müfredatına uygun" talimatı Claude'un güncel
# bilgisini tetikler.
SUBJECT_TOPICS: dict[str, dict[int, list[str]]] = {
    "matematik": {
        9: [
            "kümeler",
            "denklemler ve eşitsizlikler",
            "mutlak değer",
            "üslü ifadeler",
            "üçgenler",
        ],
        10: [
            "fonksiyonlar",
            "polinomlar",
            "ikinci dereceden denklemler",
            "dörtgenler",
            "olasılık",
        ],
        11: [
            "logaritma",
            "diziler",
            "limit ve süreklilik",
            "türev uygulamaları",
            "analitik geometri",
        ],
        12: [
            "integral",
            "trigonometri",
            "karmaşık sayılar",
            "matrisler",
            "istatistik ve veri analizi",
        ],
    },
    "fizik": {
        9: [
            "fizik bilimine giriş",
            "madde ve özellikleri",
            "hareket",
            "kuvvet",
            "enerji",
        ],
        10: [
            "elektrik ve manyetizma",
            "basınç ve kaldırma kuvveti",
            "dalgalar",
            "optik",
        ],
        11: [
            "vektörler",
            "kuvvet ve hareket",
            "elektriksel kuvvet ve alan",
            "manyetizma",
        ],
        12: [
            "düzgün çembersel hareket",
            "basit harmonik hareket",
            "dalga mekaniği",
            "atom fiziği",
        ],
    },
    "kimya": {
        9: [
            "atom modelleri",
            "periyodik sistem",
            "kimyasal türler arası etkileşimler",
            "maddenin halleri",
        ],
        10: ["karışımlar", "asitler ve bazlar", "tuzlar", "kimya her yerde"],
        11: [
            "modern atom teorisi",
            "gazlar",
            "sıvı çözeltiler",
            "kimyasal tepkimelerde enerji",
        ],
        12: [
            "kimya ve elektrik",
            "karbon kimyası",
            "organik bileşikler",
            "enerji kaynakları",
        ],
    },
    "biyoloji": {
        9: [
            "yaşam bilimi",
            "hücre",
            "canlıların çeşitliliği",
            "canlılarda enerji dönüşümleri",
        ],
        10: ["hücre bölünmeleri", "kalıtımın genel ilkeleri", "ekosistem ekolojisi"],
        11: [
            "insan fizyolojisi",
            "denetleyici ve düzenleyici sistemler",
            "duyu organları",
        ],
        12: [
            "genden proteine",
            "biyoteknoloji ve gen mühendisliği",
            "komünite ve popülasyon ekolojisi",
        ],
    },
    "tarih": {
        9: [
            "İslam öncesi Türk devletleri",
            "İlk Çağ uygarlıkları",
            "İlk Türk-İslam devletleri",
        ],
        10: ["Osmanlı kuruluş dönemi", "Osmanlı yükseliş dönemi", "Avrupa'da yeni çağ"],
        11: [
            "Osmanlı duraklama ve gerileme",
            "Tanzimat dönemi",
            "I. ve II. Meşrutiyet",
        ],
        12: [
            "Kurtuluş Savaşı",
            "Atatürk ilke ve inkılapları",
            "Soğuk Savaş",
            "Çağdaş Türkiye tarihi",
        ],
    },
    "cografya": {
        9: [
            "coğrafyaya giriş",
            "harita bilgisi",
            "iklim ve hava olayları",
            "yer şekilleri",
        ],
        10: [
            "nüfus ve göç",
            "ekonomik faaliyetler",
            "Türkiye'nin konumu ve özellikleri",
        ],
        11: ["biyoçeşitlilik", "doğal afetler", "uluslararası ulaşım ağları"],
        12: [
            "bölgesel kalkınma",
            "çevre ve toplum",
            "küresel sorunlar ve çözüm yolları",
        ],
    },
    "felsefe": {
        # MEB müfredatında felsefe 10. sınıftan itibaren. 9'da yok.
        10: ["felsefeye giriş", "bilgi felsefesi", "bilim felsefesi"],
        11: ["varlık felsefesi", "ahlak felsefesi", "siyaset felsefesi"],
        12: ["sanat felsefesi", "din felsefesi", "çağdaş felsefe akımları"],
    },
    "edebiyat": {
        9: ["edebiyata giriş", "hikaye türü", "şiir türü", "destan ve masal"],
        10: ["İslamiyet öncesi Türk edebiyatı", "Divan edebiyatı", "Halk edebiyatı"],
        11: ["Tanzimat edebiyatı", "Servet-i Fünun edebiyatı", "Milli Edebiyat dönemi"],
        12: [
            "Cumhuriyet dönemi Türk edebiyatı",
            "roman",
            "tiyatro",
            "çağdaş Türk şairleri",
        ],
    },
    "ingilizce": {
        9: [
            "present/past simple tenses",
            "describing people and places",
            "daily routines",
            "countries and nationalities",
        ],
        10: ["modals", "past continuous", "future plans", "reported speech basics"],
        11: ["perfect tenses", "passive voice", "conditionals", "phrasal verbs"],
        12: ["advanced grammar", "academic writing", "idioms", "literary analysis"],
    },
}


def build_prompt(subject: str, grade: int, count: int) -> str:
    """MEB müfredat odaklı Q&A üretimi için Claude prompt'u oluştur.

    İki yerde MEB uyumluluk zorlanır:
    1. System persona (kullanıcı rolü)
    2. Kurallar listesinin ilk maddesi
    """
    topics = SUBJECT_TOPICS[subject].get(grade)
    if not topics:
        # Bu subject × grade kombinasyonu müfredatta mevcut değil (örn. felsefe 9).
        return ""

    topics_str = ", ".join(topics)
    return (
        "Sen Türkiye Milli Eğitim Bakanlığı (MEB) güncel öğretim programlarına "
        "hakim, Türkçe lise eğitimi için içerik üreten bir asistansın.\n\n"
        f"Görev: {grade}. sınıf {subject} dersi için {count} adet "
        "soru-cevap çifti üret.\n\n"
        f"MEB müfredat konuları (bu konulardan dengeli şekilde seç): {topics_str}\n\n"
        "Kurallar:\n"
        "1. Sorular MEB güncel müfredatı ile uyumlu, sınıf seviyesine uygun olmalı.\n"
        "2. Cevaplar en az 3 cümle, pedagojik tonla (öğretici, adım adım açıklayıcı).\n"
        "3. Cevaplar 1000 karakteri aşmamalı.\n"
        "4. Emin olmadığın tarih/isim/formül varsa daha genel ifade kullan — "
        "uydurulmuş bilgi (hallucination) yazma.\n"
        "5. Soru tiplerini çeşitlendir: tanım soruları, neden-sonuç, problem çözme, "
        "karşılaştırma, örnekleme. Tekrara düşme.\n\n"
        "Çıktı formatı: SADECE JSON array. Markdown kod bloğu, açıklama veya "
        "başka metin ekleme.\n"
        'Örnek: [{"instruction": "...", "output": "..."}, ...]'
    )


def extract_json_array(text: str) -> str:
    """Claude response'undan JSON array'i ayıkla.

    Claude bazen markdown code fence içine sarıyor (``` ... ```) veya
    başa/sona açıklama metni ekliyor. İlk '[' ile son ']' arasını al.
    """
    # Markdown code fence'i yakalamaya çalış
    fence = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text)
    if fence:
        return fence.group(1)
    # Düz array: ilk '[' ile son ']' arası
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text  # parse aşamasında hata verecek — exception retry'a yol açar


def call_claude(
    client: anthropic.Anthropic,
    model: str,
    prompt: str,
    max_tokens: int = 4096,
    max_retries: int = 3,
) -> list[dict]:
    """Claude'a prompt at, JSON array olarak parse et.

    Hata türlerine göre farklı retry stratejisi:
      - Parse hatası (bozuk JSON)     → hemen yeniden dene (max 3)
      - RateLimitError                → exponential backoff
      - API hatası                    → 1s bekle + yeniden dene
    """
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            message = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            text = message.content[0].text
            parsed = json.loads(extract_json_array(text))
            if not isinstance(parsed, list):
                raise ValueError("response beklenen JSON array değil")
            return parsed
        except (json.JSONDecodeError, ValueError, KeyError, IndexError) as err:
            last_error = err
            print(
                f"    [uyarı] Parse hatası (deneme {attempt}/{max_retries}): {err}",
                file=sys.stderr,
            )
        except anthropic.RateLimitError as err:
            last_error = err
            wait = 2**attempt
            print(
                f"    [uyarı] Rate limit (deneme {attempt}/{max_retries}), "
                f"{wait}s bekleniyor",
                file=sys.stderr,
            )
            time.sleep(wait)
        except anthropic.APIError as err:
            last_error = err
            print(
                f"    [uyarı] API hatası (deneme {attempt}/{max_retries}): {err}",
                file=sys.stderr,
            )
            time.sleep(1)

    print(f"    [HATA] Tüm denemeler başarısız: {last_error}", file=sys.stderr)
    return []


def generate_dataset(
    client: anthropic.Anthropic,
    model: str,
    target_per_subject: int,
    raw_path: Path,
) -> list[dict]:
    """Tüm subject × grade kombinasyonları için Claude'dan Q&A topla.

    Ham response'lar raw_path'e akıtılır (debug + retry analizi için).
    Subject başına hedef `target_per_subject` grade sayısına göre bölünür;
    her grade için minimum 5 örnek garantisi.
    """
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    with raw_path.open("w", encoding="utf-8") as raw_file:
        for subject, grade_topics in SUBJECT_TOPICS.items():
            # Subject için hangi grade'ler müfredatta? Hedefi o sayıya böl.
            # Örn. felsefe (3 grade) → 50/3 = 16; matematik (4 grade) → 50/4 = 12.
            grades_available = list(grade_topics.keys())
            per_grade = max(5, target_per_subject // len(grades_available))

            for grade in grades_available:
                prompt = build_prompt(subject, grade, per_grade)
                if not prompt:
                    continue

                print(f"  → {subject} / grade {grade} / {per_grade} örnek...")
                items = call_claude(client, model, prompt)

                for item in items:
                    # Defensive parsing — Claude bazen ekstra alan ekliyor.
                    if not isinstance(item, dict):
                        continue
                    instruction = str(item.get("instruction", "")).strip()
                    output = str(item.get("output", "")).strip()
                    if not instruction or not output:
                        continue

                    record = {
                        "instruction": instruction,
                        "input": "",  # Alpaca uyumluluğu için boş string
                        "output": output,
                        "subject": subject,
                        "grade": grade,
                    }
                    records.append(record)
                    # Ham log — filtrelemeden önce her kayıt
                    raw_file.write(json.dumps(record, ensure_ascii=False) + "\n")

                # Nazik rate limiting — Claude API tier'ını zorlamama
                time.sleep(0.5)

    return records


def apply_quality_filters(
    records: list[dict],
    min_output_len: int = 20,
    max_output_len: int = 1000,
) -> tuple[list[dict], dict[str, int]]:
    """Uzunluk + exact duplicate instruction filtreleri.

    Returns:
        (geçen kayıtlar, her filtre türünün elediği sayı)
    """
    stats = {"length": 0, "exact_duplicate": 0}
    seen_instructions: set[str] = set()
    clean: list[dict] = []

    for rec in records:
        out_len = len(rec["output"])
        if out_len < min_output_len or out_len > max_output_len:
            stats["length"] += 1
            continue

        # Normalize et (whitespace + lowercase) — yüzeysel dedup
        key = " ".join(rec["instruction"].lower().split())
        if key in seen_instructions:
            stats["exact_duplicate"] += 1
            continue
        seen_instructions.add(key)
        clean.append(rec)

    return clean, stats


def apply_semantic_dedup(
    records: list[dict],
    threshold: float = 0.95,
) -> tuple[list[dict], int]:
    """Opsiyonel: sentence-transformers ile semantic duplicate tespiti.

    Pair-wise cosine similarity > threshold olan instruction çiftlerinden
    ikincisini eler. O(N²) karmaşıklık — ~500 örnek için hızlı; 5000+ için
    FAISS index'leme gerekir.
    """
    try:
        from sentence_transformers import SentenceTransformer, util
    except ImportError:
        print("  [uyarı] sentence-transformers kurulu değil, semantic dedup atlanıyor")
        return records, 0

    print("  → Semantic dedup modeli yükleniyor...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    instructions = [r["instruction"] for r in records]
    embeddings = model.encode(
        instructions, convert_to_tensor=True, show_progress_bar=False
    )
    similarity = util.cos_sim(embeddings, embeddings)

    # Üst üçgeni tara (diagonal hariç); i < j ve similarity > threshold → j'yi ele
    to_remove: set[int] = set()
    n = len(records)
    for i in range(n):
        if i in to_remove:
            continue
        for j in range(i + 1, n):
            if j in to_remove:
                continue
            if similarity[i][j].item() > threshold:
                to_remove.add(j)

    kept = [rec for idx, rec in enumerate(records) if idx not in to_remove]
    return kept, len(to_remove)


def stratified_split(
    records: list[dict],
    eval_ratio: float = 0.2,
) -> tuple[list[dict], list[dict]]:
    """Subject × grade kombinasyonunu koruyarak train/eval split.

    sklearn her stratum için en az 2 örnek bekler. Bir kombinasyonda tek
    örnek varsa fallback olarak sadece subject'e stratify'a düşüyoruz.
    """
    strata = [f"{r['subject']}_{r['grade']}" for r in records]
    counter = Counter(strata)

    try:
        if min(counter.values()) >= 2:
            return train_test_split(
                records,
                test_size=eval_ratio,
                stratify=strata,
                random_state=SEED,
            )
        # Fallback 1: sadece subject'e stratify
        print(
            "  [uyarı] Bazı subject_grade grupları tek örnek içeriyor; "
            "sadece subject'e göre stratify'a geçiliyor"
        )
        return train_test_split(
            records,
            test_size=eval_ratio,
            stratify=[r["subject"] for r in records],
            random_state=SEED,
        )
    except ValueError as err:
        # Fallback 2: tamamen rastgele (son çare)
        print(f"  [uyarı] Stratified split başarısız ({err}); rastgele split")
        return train_test_split(records, test_size=eval_ratio, random_state=SEED)


def write_jsonl(records: list[dict], path: Path) -> None:
    """UTF-8 JSONL yazımı — her satır bir JSON objesi."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def print_statistics(
    records: list[dict],
    filter_stats: dict[str, int],
    semantic_removed: int,
    train_count: int,
    eval_count: int,
) -> None:
    """Dataset özeti + subject × grade heatmap + uzunluk istatistikleri."""
    print("\n" + "=" * 60)
    print("Dataset Özeti")
    print("=" * 60)
    print(f"Quality filter (length):  {filter_stats['length']:>4} elenen")
    print(f"Exact duplicate:          {filter_stats['exact_duplicate']:>4} elenen")
    print(f"Semantic duplicate:       {semantic_removed:>4} elenen")
    print(f"Final toplam:             {len(records):>4}")
    print(f"Train / Eval:             {train_count} / {eval_count}")

    # Subject × grade dağılımı
    print("\nDağılım (subject × grade)")
    print("-" * 60)
    dist: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for rec in records:
        dist[rec["subject"]][rec["grade"]] += 1

    grades_sorted = sorted({rec["grade"] for rec in records})
    header = (
        f"{'subject':<15}" + "".join(f"{g:>6}" for g in grades_sorted) + f"{'Total':>8}"
    )
    print(header)
    for subject in sorted(dist):
        row = f"{subject:<15}"
        total = 0
        for g in grades_sorted:
            count = dist[subject].get(g, 0)
            row += f"{count:>6}"
            total += count
        row += f"{total:>8}"
        print(row)

    # Output uzunluk istatistikleri — karakter bazlı.
    # Token bazlı analiz Task 5 notebook'unda tokenizer ile yapılacak.
    lengths = [len(rec["output"]) for rec in records]
    print("\nOutput uzunluğu (karakter)")
    print("-" * 60)
    print(f"Ortalama: {mean(lengths):>6.0f}")
    print(f"Medyan:   {median(lengths):>6.0f}")
    print(f"Min/Max:  {min(lengths):>6} / {max(lengths)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EduAI P2 — Türkçe lise Q&A veri üretici (MEB müfredat uyumlu)"
    )
    parser.add_argument(
        "--target",
        type=int,
        default=50,
        help="Subject başına hedef örnek sayısı (default: 50)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=list(MODEL_ALIASES.keys()),
        help=f"Claude model alias'ı (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--semantic-dedup",
        action="store_true",
        help="sentence-transformers ile ek semantic deduplication",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="API call yapma; mock dataset ile pipeline doğrulama",
    )
    return parser.parse_args()


def mock_records() -> list[dict]:
    """--dry-run için deterministik mock dataset (pipeline testi)."""
    out: list[dict] = []
    for subject in list(SUBJECT_TOPICS.keys())[:4]:
        for grade in [9, 10, 11, 12]:
            if grade not in SUBJECT_TOPICS[subject]:
                continue
            for i in range(6):
                out.append(
                    {
                        "instruction": f"{subject} dersi sınıf {grade} mock sorusu {i}",
                        "input": "",
                        "output": (
                            f"{subject} / {grade}. sınıf için pedagojik bir cevap örneği. "
                            "Bu mock veri yalnızca pipeline doğrulamak içindir. "
                            "Gerçek üretimde Claude API cevapları çok daha zengindir."
                        ),
                        "subject": subject,
                        "grade": grade,
                    }
                )
    return out


def main() -> None:
    args = parse_args()
    random.seed(SEED)

    print(f"EduAI P2 — Veri üretimi (seed={SEED})")
    print("  Strateji: Seçenek A — Sentetik (Claude API)")
    print(f"  Model:    {MODEL_ALIASES[args.model]}")
    print(f"  Hedef:    {args.target} örnek/subject")
    print()

    if args.dry_run:
        print("[DRY-RUN] Mock dataset kullanılıyor (API call yok)\n")
        records = mock_records()
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(records, RAW_PATH)
    else:
        load_dotenv(ML_ROOT / ".env")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print(
                "[HATA] ANTHROPIC_API_KEY .env'de bulunamadı. "
                ".env.example'ı ml/.env olarak kopyala ve anahtarı ekle.",
                file=sys.stderr,
            )
            sys.exit(1)

        client = anthropic.Anthropic(api_key=api_key)
        print("Claude API çağrıları başlıyor (tahmini süre: 5-10 dakika)...\n")
        records = generate_dataset(
            client=client,
            model=MODEL_ALIASES[args.model],
            target_per_subject=args.target,
            raw_path=RAW_PATH,
        )
        print(f"\nHam veri üretildi: {len(records)} kayıt")

    # --- Filtre pipeline ---
    print("\n→ Kalite filtreleri uygulanıyor...")
    records, filter_stats = apply_quality_filters(records)

    semantic_removed = 0
    if args.semantic_dedup:
        print("→ Semantic dedup uygulanıyor...")
        records, semantic_removed = apply_semantic_dedup(records)

    if len(records) < 10:
        print(
            f"[HATA] Filtreleme sonrası yalnızca {len(records)} kayıt kaldı; "
            "split için yetersiz.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("\n→ Stratified %80/%20 train/eval split...")
    train, eval_ = stratified_split(records)

    write_jsonl(train, TRAIN_PATH)
    write_jsonl(eval_, EVAL_PATH)
    print(f"\n✓ {TRAIN_PATH.relative_to(ML_ROOT.parent)}")
    print(f"✓ {EVAL_PATH.relative_to(ML_ROOT.parent)}")

    print_statistics(records, filter_stats, semantic_removed, len(train), len(eval_))


if __name__ == "__main__":
    main()
