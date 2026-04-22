"""
EduAI P2 — QLoRA fine-tuning ana scripti.

Training loop: config.yaml + data/processed/*.jsonl → LoRA adapter.
MLflow deney takibi + step-bazlı metric log + KeyboardInterrupt handling.

Kullanım:
    python training/train.py                # Full training (config.yaml defaults)
    python training/train.py --smoke        # 10 step, 1 epoch — pipeline doğrulama
    python training/train.py --config path/to/other.yaml

Çalışma ortamı:
    - **CUDA zorunlu** — QLoRA bitsandbytes'a, o da CUDA'ya bağlı.
    - macOS lokal'de çalışmaz; Colab T4/Pro A100 veya Linux GPU kullan.
    - Mac'te import syntax check (`python -c "import training.train"`) yapılabilir
      ama `python training/train.py` çağırınca anlaşılır hata verir.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import mlflow
import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

# --- Yol sabitleri ------------------------------------------------------

# ml/training/ → ml/
ML_ROOT = Path(__file__).resolve().parent.parent


# --- Yapılandırma yükleme ----------------------------------------------

def load_config(path: str | Path) -> dict:
    """YAML config'i dict'e oku. Path ml/ kökünden relative."""
    p = Path(path)
    if not p.is_absolute():
        p = ML_ROOT / p
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


# --- Ortam doğrulama ----------------------------------------------------

def require_cuda() -> None:
    """QLoRA CUDA bağımlı; yoksa açık mesajla çık."""
    if not torch.cuda.is_available():
        print(
            "[HATA] CUDA bulunamadı. QLoRA training için GPU şart.\n"
            "       macOS Apple Silicon'da QLoRA desteklenmez (bitsandbytes CUDA-only).\n"
            "       Colab T4 / Colab Pro A100 veya Linux GPU'da çalıştır.\n"
            "       Colab notebook için: docs/p2/SPEC.md § Colab workflow",
            file=sys.stderr,
        )
        sys.exit(1)


# --- Model + tokenizer kurulum -----------------------------------------

def setup_model_and_tokenizer(config: dict) -> tuple:
    """Base model'i 4-bit quantize ile yükle, tokenizer + pad_token eşitle."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config["quantization"]["load_in_4bit"],
        bnb_4bit_compute_dtype=getattr(torch, config["quantization"]["bnb_4bit_compute_dtype"]),
        bnb_4bit_quant_type=config["quantization"]["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=config["quantization"].get("bnb_4bit_use_double_quant", True),
    )

    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        quantization_config=bnb_config,
        device_map="auto",
        # T4 BF16 desteklemez (compute capability 7.5 < 8.0). Phi-3 default
        # torch_dtype="bfloat16" — override etmezsek non-quantized katmanlar
        # (embedding, layer_norm, lm_head) bf16'da kalır ve gradient scaler
        # T4'te çöker. fp16'ya zorluyoruz.
        dtype=torch.float16,             # transformers 4.57: torch_dtype → dtype
        trust_remote_code=config["model"].get("trust_remote_code", False),
    )

    # QLoRA hazırlığı — SPEC'te atlanmış ama standart pattern:
    #   - layer_norm ve output head'i fp32'ye taşır (stability)
    #   - gradient checkpointing'i aktive eder (VRAM tasarrufu)
    #   - input embedding'e require_grad hook ekler
    # Bu olmadan 4-bit quantized model üzerinde gradient akışı hatalı olur.
    model = prepare_model_for_kbit_training(model)

    # use_cache + gradient_checkpointing birlikte warning verir ve bazen
    # backward pass'i bozar. QLoRA'da cache'e ihtiyaç yok (training mode).
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"],
        trust_remote_code=config["model"].get("trust_remote_code", False),
    )
    # Bazı model'lerde pad_token tanımlı değil — eos ile eşitle
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# --- LoRA adaptörlerini model'e bağla ----------------------------------

def apply_lora(model, config: dict):
    """Base model üzerine LoRA adapter ekle; orijinal ağırlıklar donuyor."""
    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        lora_dropout=config["lora"]["lora_dropout"],
        target_modules=config["lora"]["target_modules"],
        bias=config["lora"]["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, lora_config)


# --- Prompt formatlama -------------------------------------------------

def build_formatting_func(tokenizer):
    """Tokenizer'ı closure ile kapatıp formatting_func döndür.

    tokenizer.apply_chat_template model-agnostik: Phi-3, Qwen, LLaMA hepsi
    kendi chat template'ini biliyor. Bu şekilde config'te model değişirse
    format otomatik adapte olur (SPEC'te hardcoded <|user|>... yerine).
    """
    def format_prompt(example: dict) -> str:
        messages = [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    return format_prompt


# --- Step-bazlı metric callback ----------------------------------------

class StepMetricsCallback(TrainerCallback):
    """Her logging_steps'te konsola Türkçe özet + MLflow metric log.

    HuggingFace Trainer zaten stdout'a log yazıyor; bu callback ek olarak
    okunaklı tek-satırlık özet + MLflow'a explicit metric push sağlar.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        step = state.global_step

        # Training step (loss + lr + epoch + grad_norm)
        if "loss" in logs and "learning_rate" in logs:
            print(
                f"  Step {step:>5} | Loss: {logs['loss']:.4f} | "
                f"LR: {logs['learning_rate']:.2e} | Epoch: {logs.get('epoch', 0):.2f}"
            )
            mlflow.log_metric("train_loss", logs["loss"], step=step)
            mlflow.log_metric("learning_rate", logs["learning_rate"], step=step)

        # Grad norm — overfitting/instability sinyali (düşüş tümü = healthy)
        if "grad_norm" in logs:
            mlflow.log_metric("grad_norm", logs["grad_norm"], step=step)

        # Mean token accuracy — Türkçe pedagojik token tahmin doğruluğu
        if "mean_token_accuracy" in logs:
            mlflow.log_metric("mean_token_accuracy", logs["mean_token_accuracy"], step=step)

        # Evaluation step
        if "eval_loss" in logs:
            print(f"  [eval] Step {step:>5} | Eval loss: {logs['eval_loss']:.4f}")
            mlflow.log_metric("eval_loss", logs["eval_loss"], step=step)
            if "eval_mean_token_accuracy" in logs:
                mlflow.log_metric(
                    "eval_mean_token_accuracy", logs["eval_mean_token_accuracy"], step=step
                )


# --- Trainable parametre sayısı ----------------------------------------

def log_trainable_params(model) -> None:
    """Eğitilebilir vs toplam parametre oranı — MLflow'a metric olarak."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total

    mlflow.log_metrics({
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": pct,
    })
    print(
        f"  Trainable: {trainable:,} / {total:,} ({pct:.2f}%)  "
        f"← LoRA beklenen %0.1-1 aralığında"
    )


# --- Adapter + tokenizer kaydetme --------------------------------------

def save_artifacts(trainer, tokenizer, output_dir: Path, mlflow_active: bool = True) -> None:
    """LoRA adapter (base model değil) + tokenizer'ı output_dir'e yaz.

    mlflow_active=True ise aynı artifact'leri run'a da ekler — model registry
    için referans.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # trainer.model PeftModel; save_pretrained sadece adapter ağırlıklarını yazar
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"  ✓ Adapter + tokenizer kaydedildi: {output_dir}")

    if mlflow_active:
        mlflow.log_artifacts(str(output_dir), artifact_path="lora_adapter")


# --- Ana training loop --------------------------------------------------

def train(config: dict, smoke: bool, run_name_override: str | None = None) -> None:
    """Full training pipeline: model + LoRA + SFTTrainer + MLflow wrap."""
    require_cuda()
    set_seed(config["seed"])

    # MLflow setup — tracking_uri lokal file store ya da Colab'da Drive path
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # Run adı: user override varsa onu kullan, yoksa otomatik model-r-lr
    if run_name_override:
        run_name = run_name_override
    else:
        model_short = config["model"]["name"].split("/")[-1]
        run_name = (
            f"{model_short}-r{config['lora']['r']}-lr{config['training']['learning_rate']}"
        )
    if smoke:
        run_name += "-SMOKE"

    output_dir = ML_ROOT / config["paths"]["output_dir"]

    with mlflow.start_run(run_name=run_name, tags=config["mlflow"]["run_tags"]) as run:
        # Parametreleri MLflow'a tek seferde yaz (karşılaştırma için)
        mlflow.log_params({
            "model": config["model"]["name"],
            "seed": config["seed"],
            "lora_r": config["lora"]["r"],
            "lora_alpha": config["lora"]["lora_alpha"],
            "lora_dropout": config["lora"]["lora_dropout"],
            "lora_target_modules": ",".join(config["lora"]["target_modules"]),
            "learning_rate": config["training"]["learning_rate"],
            "num_epochs": config["training"]["num_epochs"],
            "warmup_ratio": config["training"]["warmup_ratio"],
            "lr_scheduler": config["training"]["lr_scheduler_type"],
            "max_length": config["model"]["max_length"],
            "effective_batch_size": (
                config["training"]["per_device_train_batch_size"]
                * config["training"]["gradient_accumulation_steps"]
            ),
            "quantization": "4bit_nf4",
            "smoke_mode": smoke,
        })

        print(f"\n→ Model yükleniyor: {config['model']['name']}")
        model, tokenizer = setup_model_and_tokenizer(config)

        print("→ LoRA adaptörleri ekleniyor...")
        model = apply_lora(model, config)
        log_trainable_params(model)

        print("→ Dataset yükleniyor...")
        dataset = load_dataset(
            "json",
            data_files={
                "train": str(ML_ROOT / config["paths"]["train_data"]),
                "eval": str(ML_ROOT / config["paths"]["eval_data"]),
            },
        )
        print(f"  Train: {len(dataset['train'])} örnek")
        print(f"  Eval:  {len(dataset['eval'])} örnek")

        # SFTConfig — TRL 0.12+ API; eski sürümlerde processing_class yerine tokenizer=
        sft_config = SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=config["training"]["num_epochs"],
            max_steps=10 if smoke else -1,   # smoke = 10 step cap
            per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
            gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
            learning_rate=config["training"]["learning_rate"],
            warmup_ratio=config["training"]["warmup_ratio"],
            lr_scheduler_type=config["training"]["lr_scheduler_type"],
            max_grad_norm=config["training"]["max_grad_norm"],
            weight_decay=config["training"].get("weight_decay", 0.0),
            logging_steps=config["training"]["logging_steps"],
            eval_strategy="steps",
            eval_steps=config["training"]["eval_steps"],
            save_strategy="steps",
            save_steps=config["training"]["save_steps"],
            save_total_limit=config["training"]["save_total_limit"],
            load_best_model_at_end=(not smoke) and config["training"]["load_best_model_at_end"],
            metric_for_best_model=config["training"]["metric_for_best_model"],
            fp16=config["training"].get("fp16", True),
            bf16=config["training"].get("bf16", False),
            # paged_adamw_8bit — bitsandbytes canonical QLoRA optimizer'ı.
            # Kendi grad unscale'ini yapar, torch AMP foreach kernel'lerini
            # bypass eder (T4'te BF16 grad unscale kernel'i yok — hatanın kökü).
            # Ek yarar: optimizer state 8-bit → VRAM ~%30 tasarruf.
            optim=config["training"].get("optim", "paged_adamw_8bit"),
            max_length=config["model"]["max_length"],   # TRL 1.0+ yeni isim (eski: max_seq_length)
            seed=config["seed"],
            report_to="none",                # MLflow'u callback üzerinden manuel yazıyoruz
        )

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,      # TRL 0.11+ API; 0.10- için tokenizer=
            train_dataset=dataset["train"],
            eval_dataset=dataset["eval"],
            formatting_func=build_formatting_func(tokenizer),
            args=sft_config,
            callbacks=[StepMetricsCallback()],
        )

        # SIGINT/SIGTERM için graceful shutdown sinyali (KeyboardInterrupt yakalar)
        def handle_interrupt(signum, frame):
            print("\n\n[UYARI] Interrupt sinyali alındı — graceful shutdown", file=sys.stderr)
            raise KeyboardInterrupt()
        signal.signal(signal.SIGTERM, handle_interrupt)

        print(f"\n→ Training başlıyor (run: {run_name})")
        start = time.monotonic()
        interrupted = False
        try:
            train_result = trainer.train()
            final_loss = train_result.training_loss
        except KeyboardInterrupt:
            interrupted = True
            print("\n[UYARI] Training kullanıcı tarafından iptal edildi — "
                  "mevcut adapter kaydediliyor...", file=sys.stderr)
            final_loss = None

        duration = time.monotonic() - start

        # Artifact kaydetme — interrupt durumunda da çalışır (elde ne varsa kaydet)
        save_artifacts(trainer, tokenizer, output_dir)

        mlflow.log_metrics({
            "training_duration_seconds": duration,
            **({"train_loss_final": final_loss} if final_loss is not None else {}),
        })
        mlflow.log_param("interrupted", interrupted)

        print("\n" + "=" * 60)
        print(f"  Süre:        {duration:.1f}s ({duration / 60:.1f} dk)")
        print(f"  MLflow run:  {run.info.run_id}")
        print(f"  Adapter:     {output_dir}")
        print(f"  Durum:       {'INTERRUPTED' if interrupted else 'OK'}")
        print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EduAI P2 — QLoRA fine-tuning (config.yaml + CLI override)"
    )
    parser.add_argument(
        "--config", type=str, default="training/config.yaml",
        help="Yapılandırma YAML yolu (ml/ kökünden relative)",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Hızlı pipeline testi: max_steps=10",
    )

    # --- Task 3 sweep için CLI override'lar ---
    # Config dosyasını duplicate etmek yerine in-memory override:
    # ayrı varyasyonlar için aynı config + farklı CLI arg'ları.
    parser.add_argument(
        "--lora-r", type=int, default=None,
        help="Override config.lora.r (alpha otomatik r*2'ye ayarlanır)",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override config.training.learning_rate",
    )
    parser.add_argument(
        "--num-epochs", type=int, default=None,
        help="Override config.training.num_epochs",
    )
    parser.add_argument(
        "--run-name-suffix", type=str, default=None,
        help="MLflow run adı (None ise otomatik: model-r{R}-lr{LR})",
    )
    return parser.parse_args()


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """CLI argümanlarını config dict'i üzerine in-memory override uygula.

    Not — `lora_alpha` otomatik `r * 2`'ye ayarlanır. Bu scale factor
    (`alpha/r = 2.0`) sabit tutar; konvansiyonel LoRA davranışı.
    Ayrı alpha istenirse bu fonksiyona `--lora-alpha` eklenebilir.
    """
    if args.lora_r is not None:
        config["lora"]["r"] = args.lora_r
        config["lora"]["lora_alpha"] = args.lora_r * 2
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.num_epochs is not None:
        config["training"]["num_epochs"] = args.num_epochs
    return config


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = apply_cli_overrides(config, args)
    train(config, smoke=args.smoke, run_name_override=args.run_name_suffix)


if __name__ == "__main__":
    main()
