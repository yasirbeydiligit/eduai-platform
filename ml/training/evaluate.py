"""
evaluate.py — EduAI P2 fine-tuned adapter değerlendirme scripti.

Base Phi-3 + LoRA adapter yükler, eval.jsonl'den ilk N örnek için
cevap üretir, ROUGE-1/ROUGE-L + BERTScore F1 hesaplar, inference
latency ölçer, 5 örnek yan yana yazdırır ve sonuçları MLflow'a
"evaluation_" prefix ile ayrı run olarak logler.

Kullanım:
    python training/evaluate.py                       # config + son checkpoint
    python training/evaluate.py --n 50                # ilk 50 örnek
    python training/evaluate.py --adapter /path/to/x  # farklı adapter dizini
    python training/evaluate.py --run-name "eval-r16" # MLflow run adı override

Çalışma ortamı:
    - CUDA zorunlu (QLoRA inference)
    - Train ile aynı Colab T4 runtime; evaluate train sonrası aynı session'da
      veya yeni session'da yeni adapter yükler
    - BERTScore ilk çağrıda ~1.5GB xlm-roberta-large indirir (cache'li)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlflow
import torch
import yaml
from bert_score import score as bert_score_fn
from peft import PeftModel
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ml/training/ → ml/
ML_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path = "training/config.yaml") -> dict:
    """YAML config'i dict'e oku (train.py ile aynı loader)."""
    p = Path(path)
    if not p.is_absolute():
        p = ML_ROOT / p
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_adapter(config: dict, adapter_dir: Path) -> tuple:
    """Base model'i 4-bit quantize ile yükle, LoRA adapter'ı bağla.

    Inference mode — `model.eval()` ile dropout kapatılır. Train sırasında
    AMP grad scaler kaynaklı BF16 problemi burada yok (no_grad bağlamında
    scaler devreye girmez), `torch_dtype=torch.float16` güvenli.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config["quantization"]["load_in_4bit"],
        bnb_4bit_compute_dtype=getattr(torch, config["quantization"]["bnb_4bit_compute_dtype"]),
        bnb_4bit_quant_type=config["quantization"]["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=config["quantization"].get("bnb_4bit_use_double_quant", True),
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=config["model"].get("trust_remote_code", False),
    )

    # LoRA adapter'ı base model'e bağla — PeftModel ağırlıkları merge etmez,
    # forward pass'te base + adapter paralel çalışır (yine 4-bit VRAM'inde)
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()   # inference — dropout/batchnorm kapalı, deterministic

    # Tokenizer adapter dizininden; training sırasında oraya kaydedildi.
    # Dosya yoksa base model'in tokenizer'ına düş (defensive).
    tokenizer_source = (
        adapter_dir if (adapter_dir / "tokenizer_config.json").exists()
        else config["model"]["name"]
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_source),
        trust_remote_code=config["model"].get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_answer(
    model,
    tokenizer,
    instruction: str,
    max_new_tokens: int = 256,
) -> tuple[str, float]:
    """Instruction için deterministic cevap üret + süreyi ms olarak döndür.

    train.py'deki `build_formatting_func` ile aynı chat template (apply_chat_template).
    add_generation_prompt=True → modele "sıra sende yanıt ver" işareti gönderir.
    """
    messages = [{"role": "user", "content": instruction}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    start = time.monotonic()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,                     # deterministic — eval için şart
            pad_token_id=tokenizer.eos_token_id,
        )
    duration_ms = (time.monotonic() - start) * 1000

    # Üretilen kısım prompt'tan sonrası — sadece yeni token'lar
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text.strip(), duration_ms


def compute_metrics(
    generated: list[str],
    references: list[str],
) -> dict:
    """ROUGE-1, ROUGE-L, BERTScore F1 — hem per-sample hem ortalama."""
    # use_stemmer=False — Türkçe için güvenilir stemmer rouge-score pkg'de yok;
    # BERTScore zaten semantik örtüşmeyi morfolojiden bağımsız yakalıyor
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)

    rouge1_scores: list[float] = []
    rougeL_scores: list[float] = []
    for ref, gen in zip(references, generated, strict=True):
        s = scorer.score(ref, gen)
        rouge1_scores.append(s["rouge1"].fmeasure)
        rougeL_scores.append(s["rougeL"].fmeasure)

    # BERTScore lang="tr" — xlm-roberta-large üzerinden hesaplar
    # İlk çağrıda internet'ten ~1.5GB indirilir, sonrasında HF cache'li
    print("  → BERTScore hesaplanıyor (ilk çağrıda model indirilir)...")
    _, _, bert_f1 = bert_score_fn(
        generated, references, lang="tr", verbose=False, batch_size=8,
    )
    bert_f1_scores: list[float] = bert_f1.tolist()

    return {
        "rouge1_scores": rouge1_scores,
        "rougeL_scores": rougeL_scores,
        "bertscore_f1_scores": bert_f1_scores,
        "rouge1_mean": sum(rouge1_scores) / len(rouge1_scores),
        "rougeL_mean": sum(rougeL_scores) / len(rougeL_scores),
        "bertscore_f1_mean": sum(bert_f1_scores) / len(bert_f1_scores),
    }


def interpret_scores(rouge_l_mean: float, bert_f1_mean: float) -> list[str]:
    """Türkçe için kalibre edilmiş eşik yorumları.

    Eşikler CONCEPT.md § 8 Evaluation bölümünden — Türkçe morfolojisi
    ROUGE'u şişirmez; BERTScore daha güvenilir semantik sinyal verir.
    """
    notes: list[str] = []

    if rouge_l_mean < 0.25:
        notes.append(
            f"⚠ ROUGE-L {rouge_l_mean:.3f} < 0.25: adapter yeterince öğrenmemiş. "
            "Daha fazla epoch veya dataset büyütme gerekli."
        )
    elif rouge_l_mean < 0.40:
        notes.append(
            f"• ROUGE-L {rouge_l_mean:.3f} makul aralıkta (0.25-0.40). "
            "Manuel QA doğrulamaya bağlı."
        )
    else:
        notes.append(
            f"✓ ROUGE-L {rouge_l_mean:.3f} > 0.40: iyi ama overfitting "
            "kontrolü yap (train/eval instruction overlap var mı?)."
        )

    if bert_f1_mean < 0.60:
        notes.append(
            f"⚠ BERTScore F1 {bert_f1_mean:.3f} < 0.60: semantik örtüşme zayıf. "
            "Base model muhtemelen daha iyi — fine-tuning kalite getirmedi."
        )
    elif bert_f1_mean < 0.75:
        notes.append(
            f"• BERTScore F1 {bert_f1_mean:.3f} orta (0.60-0.75). "
            "Kabul edilebilir, manuel QA ile doğrula."
        )
    else:
        notes.append(
            f"✓ BERTScore F1 {bert_f1_mean:.3f} > 0.75: semantik örtüşme iyi."
        )

    return notes


def print_samples(
    instructions: list[str],
    references: list[str],
    generated: list[str],
    rouge_l_scores: list[float],
    n: int = 5,
) -> None:
    """İlk n örneği yan yana yazdır — manuel kalite gözlemi için."""
    print("\n" + "=" * 78)
    print(f"Örnek Çıktılar (ilk {n})")
    print("=" * 78)
    for i in range(min(n, len(instructions))):
        print(f"\n--- Örnek {i + 1} ---")
        print(f"Soru:      {instructions[i]}")
        ref_preview = references[i][:250] + ("..." if len(references[i]) > 250 else "")
        gen_preview = generated[i][:250] + ("..." if len(generated[i]) > 250 else "")
        print(f"Referans:  {ref_preview}")
        print(f"Üretilen:  {gen_preview}")
        print(f"ROUGE-L:   {rouge_l_scores[i]:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EduAI P2 — Fine-tuned adapter evaluation (ROUGE + BERTScore)"
    )
    parser.add_argument(
        "--config", type=str, default="training/config.yaml",
        help="Yapılandırma YAML (ml/ kökünden relative)",
    )
    parser.add_argument(
        "--adapter", type=str, default=None,
        help="Adapter dizini (default: config.paths.output_dir)",
    )
    parser.add_argument(
        "--n", type=int, default=20,
        help="eval.jsonl'den değerlendirilecek örnek sayısı (default 20)",
    )
    parser.add_argument(
        "--run-name", type=str, default="evaluation",
        help="MLflow evaluation run adı (default: 'evaluation')",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        print(
            "[HATA] CUDA gerekli. QLoRA inference Colab T4 veya Linux GPU'da çalışır.",
            file=sys.stderr,
        )
        sys.exit(1)

    config = load_config(args.config)

    # Adapter yolu — CLI arg > config default
    adapter_dir = (
        Path(args.adapter) if args.adapter
        else (ML_ROOT / config["paths"]["output_dir"])
    )
    if not adapter_dir.exists():
        print(f"[HATA] Adapter dizini yok: {adapter_dir}", file=sys.stderr)
        sys.exit(1)

    eval_path = ML_ROOT / config["paths"]["eval_data"]
    if not eval_path.exists():
        print(f"[HATA] Eval data yok: {eval_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Adapter:      {adapter_dir}")
    print(f"Eval data:    {eval_path}")
    print(f"Örnek sayısı: {args.n}\n")

    print("→ Base model + LoRA adapter yükleniyor...")
    model, tokenizer = load_adapter(config, adapter_dir)

    # Eval data — ilk n örnek (deterministic, seed zaten aynı)
    records = [
        json.loads(line)
        for line in eval_path.read_text(encoding="utf-8").splitlines()
    ]
    samples = records[: args.n]
    instructions = [s["instruction"] for s in samples]
    references = [s["output"] for s in samples]

    # Inference döngüsü — progress log'u her 5 sample'da
    print(f"\n→ {args.n} örnek için cevap üretiliyor...")
    generated: list[str] = []
    durations_ms: list[float] = []
    for i, instruction in enumerate(instructions, 1):
        text, dur = generate_answer(model, tokenizer, instruction)
        generated.append(text)
        durations_ms.append(dur)
        if i % 5 == 0 or i == len(instructions):
            running_avg = sum(durations_ms) / len(durations_ms)
            print(f"  [{i}/{len(instructions)}] rolling avg latency: {running_avg:.0f} ms")

    avg_latency_ms = sum(durations_ms) / len(durations_ms)

    print("\n→ Metric hesaplanıyor...")
    metrics = compute_metrics(generated, references)

    # 5 örnek yan yana — manuel göz denetimi için
    print_samples(
        instructions, references, generated, metrics["rougeL_scores"], n=5
    )

    # Özet
    print("\n" + "=" * 78)
    print("Evaluation Özeti")
    print("=" * 78)
    print(f"ROUGE-1 (avg):       {metrics['rouge1_mean']:.4f}")
    print(f"ROUGE-L (avg):       {metrics['rougeL_mean']:.4f}")
    print(f"BERTScore F1 (tr):   {metrics['bertscore_f1_mean']:.4f}")
    print(f"Avg inference:       {avg_latency_ms:.0f} ms")

    print("\nYorum (Türkçe eşikleri):")
    for note in interpret_scores(metrics["rougeL_mean"], metrics["bertscore_f1_mean"]):
        print(f"  {note}")

    # MLflow — ayrı evaluation run; training run'ları kirletmez
    print("\n→ MLflow'a logleniyor...")
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name=args.run_name, tags={"type": "evaluation"}) as run:
        mlflow.log_params({
            "adapter_path": str(adapter_dir),
            "eval_sample_count": args.n,
            "model": config["model"]["name"],
        })
        mlflow.log_metrics({
            "evaluation_rouge1": metrics["rouge1_mean"],
            "evaluation_rougeL": metrics["rougeL_mean"],
            "evaluation_bertscore_f1": metrics["bertscore_f1_mean"],
            "evaluation_avg_inference_ms": avg_latency_ms,
        })
        print(f"✓ MLflow evaluation run: {run.info.run_id}")


if __name__ == "__main__":
    main()
