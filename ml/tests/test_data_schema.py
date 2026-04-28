"""
JSONL dataset schema validation — data_prep.py çıktısı için doğrulama.

Kontrol edilenler:
    - Her kayıtta tüm required key'ler var
    - subject valid enum'da
    - grade 9-12 (P2 lise scope'u)
    - output uzunluğu data_prep.py filtreleriyle tutarlı (20-1000 char)
    - instruction ve output boş değil
    - input string (alpaca uyumluluğu)
    - Dataset içinde exact duplicate yok
    - Train ⟂ eval (leakage yok)

CI senaryosunda dataset dosyaları yoksa testler skip edilir —
data_prep.py henüz çalıştırılmamışsa false-positive fail'i engeller.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ml/tests/ → ml/
ML_ROOT = Path(__file__).resolve().parent.parent

# Geçerli ders listesi. SPEC'teki listeye sadık — "din" ve "genel"
# şu an SUBJECT_TOPICS'te yok ama ileride eklenirse geçerli kalsın.
VALID_SUBJECTS = {
    "matematik",
    "fizik",
    "kimya",
    "biyoloji",
    "tarih",
    "cografya",
    "felsefe",
    "din",
    "edebiyat",
    "ingilizce",
    "genel",
}

REQUIRED_KEYS = {"instruction", "input", "output", "subject", "grade"}

# P2 lise odaklı; data_prep.py 9-12 üretiyor.
MIN_GRADE = 9
MAX_GRADE = 12

# data_prep.py:apply_quality_filters ile aynı sınırlar.
MIN_OUTPUT_LEN = 20
MAX_OUTPUT_LEN = 1000


def _load_records(path: Path) -> list[dict]:
    """JSONL dosyasını kayıt listesine oku."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _normalize(instruction: str) -> str:
    """data_prep.py ile aynı normalization (whitespace + lowercase)."""
    return " ".join(instruction.lower().split())


@pytest.mark.parametrize("filename", ["train.jsonl", "eval.jsonl"])
def test_jsonl_schema(filename: str) -> None:
    """Her satır JSONL schema'sına uyuyor mu."""
    path = ML_ROOT / "data" / "processed" / filename
    if not path.exists():
        pytest.skip(f"{path} yok — data_prep.py henüz çalışmadı")

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) > 0, f"{filename} boş"

    for i, line in enumerate(lines, 1):
        record = json.loads(line)

        # Required keys — eksik varsa hangileri eksik listele
        missing = REQUIRED_KEYS - record.keys()
        assert not missing, f"{filename}:{i} eksik anahtarlar: {missing}"

        # Subject enum
        assert record["subject"] in VALID_SUBJECTS, (
            f"{filename}:{i} geçersiz subject: {record['subject']!r}"
        )

        # Grade range
        assert MIN_GRADE <= record["grade"] <= MAX_GRADE, (
            f"{filename}:{i} grade {MIN_GRADE}-{MAX_GRADE} dışı: {record['grade']}"
        )

        # Output uzunluğu (data_prep.py filtresiyle tutarlı)
        out_len = len(record["output"])
        assert MIN_OUTPUT_LEN <= out_len <= MAX_OUTPUT_LEN, (
            f"{filename}:{i} output uzunluğu sınır dışı: {out_len} karakter"
        )

        # Boş olmayan alanlar
        assert record["instruction"].strip(), f"{filename}:{i} boş instruction"
        assert record["output"].strip(), f"{filename}:{i} boş output"
        assert isinstance(record["input"], str), (
            f"{filename}:{i} input string olmalı (alpaca uyumluluğu)"
        )


def test_no_exact_duplicate_instructions() -> None:
    """Train + eval birleşiminde exact duplicate instruction olmamalı."""
    train_path = ML_ROOT / "data" / "processed" / "train.jsonl"
    eval_path = ML_ROOT / "data" / "processed" / "eval.jsonl"

    if not (train_path.exists() and eval_path.exists()):
        pytest.skip("data_prep.py henüz çalışmadı")

    all_keys: list[str] = []
    for path in (train_path, eval_path):
        for rec in _load_records(path):
            all_keys.append(_normalize(rec["instruction"]))

    duplicates = len(all_keys) - len(set(all_keys))
    assert duplicates == 0, f"{duplicates} exact duplicate instruction bulundu"


def test_train_eval_no_leakage() -> None:
    """Train'deki bir instruction eval'de olmamalı (data leakage koruması)."""
    train_path = ML_ROOT / "data" / "processed" / "train.jsonl"
    eval_path = ML_ROOT / "data" / "processed" / "eval.jsonl"

    if not (train_path.exists() and eval_path.exists()):
        pytest.skip("data_prep.py henüz çalışmadı")

    train_keys = {_normalize(r["instruction"]) for r in _load_records(train_path)}
    eval_keys = {_normalize(r["instruction"]) for r in _load_records(eval_path)}

    overlap = train_keys & eval_keys
    assert not overlap, (
        f"Train-eval overlap: {len(overlap)} ortak instruction. Örnek: "
        f"{next(iter(overlap))!r}"
    )
