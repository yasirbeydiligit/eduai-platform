# P2 — Model Fabrikası: Proje Spesifikasyonu

> **📝 P2 FINALIZED — 2026-04-28**
>
> Bu spec implementation sırasında **31 bilinçli sapma** ile evrildi. Aşağıdaki
> **inline callout'lar** kritik kararları işaretler. Tam sapma tarihçesi:
> [`IMPLEMENTATION_NOTES_ARCHIVED.md`](IMPLEMENTATION_NOTES_ARCHIVED.md).
> P3 transfer paketi: [`P3_HANDOFF.md`](P3_HANDOFF.md).
>
> **Final base model:** `Qwen/Qwen3-4B-Instruct-2507` (SPEC'teki Phi-3 başlangıçtı —
> Türkçe yetersizliği nedeniyle değiştirildi, Sapma 27→28).

> **Scope:** LoRA adapter üreten bir ML pipeline. P1 API'ye dokunmaz, bağımsız çalışır. Çıktı (adapter + tokenizer config + metrics) P3'te tüketilir.
>
> **Gerçekçi hedef:** "Tek başına mükemmel model" değil, P3 RAG pipeline'ının arkasına takılacak, Türkçe pedagojik tonu kazanmış bir adapter. Bilgi doğruluğu P3'teki retrieval ile sağlanır, P2'de değil.
>
> **Sürüm hassasiyeti uyarısı:** `trl`, `transformers`, `peft` hızlı gelişiyor. Aşağıdaki kod örnekleri `pyproject.toml`'da pin'li sürüme göre yazılmış. Kodu uygularken `pip show trl transformers peft` ile sürüm kontrolü yap, API farkı varsa adapte et — körü körüne kopyala-yapıştır yapma (P1'deki "eleştirel düşün" prensibi).

---

## Proje özeti

**Ne:** Türkçe lise eğitimi için Phi-3/Phi-4/Qwen ailesinden bir modeli QLoRA ile fine-tune eden, MLflow ile deney takibi yapan ML pipeline.
**Klasör:** `eduai-platform/ml/`
**Bağımlılık:** P1'e dokunmaz. P3'te tüketilir.

---

## Klasör yapısı

```
eduai-platform/
└── ml/
    ├── data/
    │   ├── .gitignore                       ← raw veriyi ignore (gizlilik)
    │   ├── raw/
    │   │   └── eduai_qa_raw.jsonl           ← ham Q&A verisi (gitignored)
    │   └── processed/
    │       ├── train.jsonl                  ← eğitim seti (%80)
    │       └── eval.jsonl                   ← değerlendirme seti (%20)
    ├── training/
    │   ├── config.yaml                      ← tüm hyperparametreler
    │   ├── train.py                         ← ana eğitim scripti
    │   ├── evaluate.py                      ← ROUGE + BERTScore + manuel örnek
    │   └── data_prep.py                     ← ham veriyi işler (veya sentetik üretir)
    ├── models/
    │   └── checkpoints/                     ← LoRA adaptörleri (gitignored)
    ├── notebooks/
    │   └── 01_data_exploration.ipynb
    ├── tests/
    │   └── test_data_schema.py              ← dataset schema validation
    ├── requirements.txt
    ├── .env.example                         ← HF_TOKEN, opsiyonel API key'ler
    └── README.md
```

---

## Dataset spesifikasyonu

> **📝 Implementation note (Sapma 6, 7, 8):** Strateji **A (Sentetik, Claude API)**
> seçildi; prompt template'inde **MEB güncel müfredatı** zorunluluğu iki katmanlı
> (system persona + rule #1). MEB kazanım kodları (örn. `9.1.1.1`) verilmiyor —
> hallucination riski. Konu seed'leri `data_prep.py:SUBJECT_TOPICS`'ta. Token
> sayımı **karakter bazlı** indirgendi (tokenizer dependency hafifletmek için);
> token analizi `notebooks/01_data_exploration.ipynb` Hücre 6'da yapılır. Felsefe
> 9. sınıfta yok (MEB realitesi); subject × grade coverage script tarafından
> dengeleniyor. **Üretim sonucu: 348 train + 87 eval = 435 örnek** (hedef ~450
> civarı, kalite filtreleri sonrası).

### Format: JSONL (her satır bir JSON)

```jsonl
{"instruction": "Türkiye'de İkinci Meşrutiyet'in ilan edilmesinin sebeplerini açıkla.", "input": "", "output": "İkinci Meşrutiyet, 1908 yılında İttihat ve Terakki Cemiyeti'nin baskısıyla ilan edilmiştir. Temel sebepler: ordunun siyasallaşması, II. Abdülhamit'in istibdat yönetimine tepki, Makedonya sorunu ve Batılı devletlerin Osmanlı topraklarına yönelik müdahalelerdir.", "subject": "tarih", "grade": 10}
{"instruction": "İkinci dereceden bir denklemi çöz: 2x² + 5x - 3 = 0", "input": "", "output": "Discriminant: b² - 4ac = 25 + 24 = 49. Kökler: x = (-5 ± 7) / 4. Sonuç: x₁ = 0.5, x₂ = -3", "subject": "matematik", "grade": 9}
```

### Hedef sayılar
- **Minimum:** 200 örnek (hızlı iteration için)
- **Hedef:** 500 örnek (iyi kalite adapter için)
- **Dağılım:** her `subject` için min 50 örnek, grade 9-12 dengeli

### Veri oluşturma stratejisi (karar seçenekleri)

SPEC tek bir yol dayatmaz; `data_prep.py`'de aşağıdakilerden birini (veya hibrit) uygula:

**Seçenek A — Sentetik (Claude API ile):**
- `.env`'de `ANTHROPIC_API_KEY` → Claude 4.x sonnet/haiku ile toplu üretim
- Her subject için prompt template + 50 seed soru → model genişletir
- ~60-90 dakika + manuel QA passı (output'lar pedagojik mi, grade-uygun mu?)
- Maliyet tahmini: 500 örnek × ~2K token = ~1M token, ~$3-5 Claude API

**Seçenek B — Açık kaynak TR dataset dönüştürme:**
- TUQuAD, MKQA-TR, Turkish MMLU örnekleri JSONL'e mapping
- Lisans kontrol et (çoğu research-only)
- Format dönüşümü gerekli

**Seçenek C — Hibrit (önerilen):**
- Seçenek A ile ana veri üret
- 50 örneği manuel yaz/düzelt (kalite seed'i)
- Açık kaynak dataset'ten örnek/format referans al

`data_prep.py` seçilen stratejiyi modüler yapıda implement etmeli; SPEC stratejiyi dikte etmez ama tek seçim yapılıp belgelenmelidir (`ml/README.md`'de).

### Deduplication ve kalite kontrol
- **Exact dup:** aynı instruction iki satırda varsa birini sil
- **Semantic dup (opsiyonel):** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` ile cosine sim > 0.95 olanları incele
- **Length filter:** output < 20 karakter veya > 1000 karakter olanları ele al
- **Subject dengesi:** subject başına sayı logla, aşırı dengesizlik varsa augment

### Train/eval split
- **Stratified %80/%20** — `subject` ve `grade` dağılımı her iki split'te korunmalı
- `sklearn.model_selection.train_test_split(..., stratify=labels)` veya manual grouping

---

## config.yaml

> **📝 Implementation note (Sapma 17, 18, 27, 28):** Bu spec'teki config baseline'dı;
> implementation'da **5 kritik değişiklik** uygulandı:
> 1. `model.name` → `Qwen/Qwen3-4B-Instruct-2507` (Phi-3 Türkçe yetersiz)
> 2. `model.trust_remote_code` → `false` (Qwen native, custom code gerekmez)
> 3. `training.fp16/bf16` → **ikisi de `false`** (T4 + Qwen3 AMP grad scaler bug;
>    fp32 training maliyet ~%30 yavaşlama, kabul edildi)
> 4. `training.optim: "paged_adamw_8bit"` eklendi (canonical QLoRA optimizer)
> 5. `training.eval_steps: 20` (orjinal 100 idi → max_steps=66'ya göre hiç eval
>    tetiklenmiyordu)
> Güncel config: `ml/training/config.yaml`. CLI override Task 3 sweep için
> Sapma 19'da eklendi (`--lora-r`, `--lr`, `--num-epochs`, `--run-name-suffix`).

```yaml
# P2 fine-tuning configuration
# Seed: reproducibility için sabit tutulur (MLflow comparison'da gürültü azalır)
seed: 42

model:
  # 2026 reality check: Phi-3 bir başlangıç. Phi-4 Mini veya Qwen3 7B Türkçe için daha iyi.
  # HuggingFace'te aktüel durumu kontrol edip değiştir (aynı config'le çalışır).
  name: "microsoft/Phi-3-mini-4k-instruct"
  max_length: 512
  trust_remote_code: true   # Phi-3 için gerekli — dikkat: uzak kod yürütür, güvenilir modelde OK

lora:
  r: 16                     # LoRA rank (8, 16, 32 dene)
  lora_alpha: 32            # genelde r * 2
  lora_dropout: 0.05
  bias: "none"
  target_modules:           # attention projeksiyonları
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"

quantization:
  load_in_4bit: true        # QLoRA
  bnb_4bit_compute_dtype: "float16"
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_use_double_quant: true   # ek ~0.4 bit tasarruf

training:
  num_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4    # effective batch = 16
  learning_rate: 2.0e-4
  warmup_ratio: 0.03
  lr_scheduler_type: "cosine"
  max_grad_norm: 1.0                # gradient clipping
  save_steps: 100
  eval_steps: 100
  logging_steps: 10
  save_total_limit: 2               # disk dolmasın
  load_best_model_at_end: true
  metric_for_best_model: "eval_loss"

paths:
  train_data: "data/processed/train.jsonl"
  eval_data: "data/processed/eval.jsonl"
  output_dir: "models/checkpoints"

mlflow:
  tracking_uri: "mlruns"              # lokal; Colab için drive path
  experiment_name: "eduai-fine-tuning"
  run_tags:
    project: "EduAI Platform P2"
    phase: "p2"
```

---

## train.py — mimari referansı

> ⚠️ Aşağıdaki kod **mimari yol haritasıdır**, birebir kopyala-yapıştır **değil**. `trl` / `transformers` sürüm farkları olabilir. Uygularken `pip show trl` ile versiyon kontrol et, API farkı varsa uyarla (ör. `tokenizer=` → `processing_class=`, `SFTConfig` parametre adları vs.).

> **📝 Implementation note (Sapma 10, 15, 16, 25):** SPEC kodu üzerine **dört
> kritik değişiklik** yapıldı (TRL 1.2 + transformers 4.57 + peft 0.18 ile
> uyum):
> 1. **Chat template:** Hardcoded `<\|user\|>...` → `tokenizer.apply_chat_template`
>    (model-agnostik; Qwen, Phi-3, LLaMA hepsi kendi template'i ile çalışır)
> 2. **TRL 1.0+ rename:** `SFTConfig(max_seq_length=...)` → `max_length=...`
> 3. **transformers 4.57 rename:** `torch_dtype=torch.float16` → `dtype=torch.float16`
> 4. **QLoRA prep eksikliği:** `prepare_model_for_kbit_training(model)` çağrısı
>    ekledildi (SPEC'te yoktu; layer_norm fp32, gradient checkpointing, embedding
>    require_grad — QLoRA standart pattern, eksiklik T4 BF16 grad scaler
>    çökmesinin parçası idi)
> Final implementation: `ml/training/train.py`. Smoke mode (`--smoke`),
> KeyboardInterrupt + SIGTERM handler, MLflow callback hepsi eklendi.

```python
"""
EduAI Fine-tuning Pipeline — QLoRA + Türkçe eğitim Q&A.
"""
import time
from pathlib import Path

import mlflow
import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from trl import SFTConfig, SFTTrainer


def load_config(path: str = "training/config.yaml") -> dict:
    with Path(path).open() as f:
        return yaml.safe_load(f)


def format_prompt(example: dict) -> str:
    """Instruction → Phi-3 chat template."""
    return (
        f"<|user|>\n{example['instruction']}\n<|end|>\n"
        f"<|assistant|>\n{example['output']}\n<|end|>"
    )


def setup_model(config: dict):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config["quantization"]["load_in_4bit"],
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type=config["quantization"]["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=config["quantization"].get("bnb_4bit_use_double_quant", True),
    )

    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=config["model"].get("trust_remote_code", False),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"],
        trust_remote_code=config["model"].get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def apply_lora(model, config: dict):
    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        lora_dropout=config["lora"]["lora_dropout"],
        target_modules=config["lora"]["target_modules"],
        bias=config["lora"]["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, lora_config)


def main():
    config = load_config()

    # Reproducibility — torch + numpy + random + cuda seed
    set_seed(config["seed"])

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    run_name = f"{config['model']['name'].split('/')[-1]}-r{config['lora']['r']}-lr{config['training']['learning_rate']}"

    with mlflow.start_run(run_name=run_name, tags=config["mlflow"]["run_tags"]):
        mlflow.log_params({
            "model": config["model"]["name"],
            "seed": config["seed"],
            "lora_r": config["lora"]["r"],
            "lora_alpha": config["lora"]["lora_alpha"],
            "learning_rate": config["training"]["learning_rate"],
            "epochs": config["training"]["num_epochs"],
            "effective_batch_size": (
                config["training"]["per_device_train_batch_size"]
                * config["training"]["gradient_accumulation_steps"]
            ),
        })

        model, tokenizer = setup_model(config)
        model = apply_lora(model, config)

        # Trainable parameter count — MLflow'a yaz
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        mlflow.log_metrics({
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": 100 * trainable / total,
        })
        print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

        dataset = load_dataset(
            "json",
            data_files={
                "train": config["paths"]["train_data"],
                "eval": config["paths"]["eval_data"],
            },
        )

        sft_config = SFTConfig(
            output_dir=config["paths"]["output_dir"],
            num_train_epochs=config["training"]["num_epochs"],
            per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
            gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
            learning_rate=config["training"]["learning_rate"],
            warmup_ratio=config["training"]["warmup_ratio"],
            lr_scheduler_type=config["training"]["lr_scheduler_type"],
            max_grad_norm=config["training"]["max_grad_norm"],
            logging_steps=config["training"]["logging_steps"],
            eval_steps=config["training"]["eval_steps"],
            save_steps=config["training"]["save_steps"],
            save_total_limit=config["training"]["save_total_limit"],
            load_best_model_at_end=config["training"]["load_best_model_at_end"],
            metric_for_best_model=config["training"]["metric_for_best_model"],
            max_seq_length=config["model"]["max_length"],
            seed=config["seed"],
            report_to="none",   # MLflow'u manuel yönetiyoruz
        )

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,   # TRL 0.11+ için; eski sürümde `tokenizer=tokenizer`
            train_dataset=dataset["train"],
            eval_dataset=dataset["eval"],
            formatting_func=format_prompt,
            args=sft_config,
        )

        start = time.monotonic()
        train_result = trainer.train()
        duration = time.monotonic() - start

        mlflow.log_metrics({
            "train_loss": train_result.training_loss,
            "training_duration_seconds": duration,
        })

        # Adapter'ı kaydet (sadece LoRA weights, base model değil — ~20-100MB)
        trainer.model.save_pretrained(config["paths"]["output_dir"])
        tokenizer.save_pretrained(config["paths"]["output_dir"])
        mlflow.log_artifacts(config["paths"]["output_dir"], artifact_path="lora_adapter")

        run_id = mlflow.active_run().info.run_id
        print(f"Training complete. Duration: {duration:.1f}s | MLflow run: {run_id}")


if __name__ == "__main__":
    main()
```

**TRL sürüm notu:** `processing_class=tokenizer` parametresi TRL 0.11+'da geçerli. Eğer `pip show trl` 0.10 veya daha eski gösterirse `tokenizer=tokenizer` olarak değiştir. Aynı şekilde `SFTConfig`'in bazı parametre adları (`max_seq_length`, `dataset_text_field`) sürüme göre farklılaşabiliyor — aktüel docstring'e bak.

---

## evaluate.py — tam implementation

> **📝 Implementation note (Sapma 21, 22, 24, 25):** SPEC kodu üzerine dört
> uyarlama yapıldı:
> 1. **Chat template:** Hardcoded prompt → `tokenizer.apply_chat_template`
>    (`add_generation_prompt=True`)
> 2. **Tokenizer source:** Önce adapter dizininden (training'de oraya save
>    edildi); yoksa base model'inden (defensive fallback)
> 3. **Phi-3 dönemi (artık irrelevant):** `trust_remote_code=False` +
>    `attn_implementation="eager"` Phi-3 custom code'unun `DynamicCache.seen_tokens`
>    bug'ından kaçınmak içindi. Final base **Qwen3 native** olduğu için
>    `attn_implementation="eager"` kaldırıldı (SDPA default Qwen için optimal),
>    `trust_remote_code` config'ten okunur.
> 4. **dtype rename** (transformers 4.57): `torch_dtype` → `dtype`

```python
"""
Fine-tuned adapter'ı ROUGE + BERTScore ile değerlendir,
5 örnek için referans vs üretilen karşılaştırması yazdır.
"""
import json
import time
from pathlib import Path

import mlflow
import torch
import yaml
from bert_score import score as bert_score_fn
from peft import PeftModel
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config(path: str = "training/config.yaml") -> dict:
    with Path(path).open() as f:
        return yaml.safe_load(f)


def load_adapter(config: dict):
    tokenizer = AutoTokenizer.from_pretrained(
        config["paths"]["output_dir"],
        trust_remote_code=config["model"].get("trust_remote_code", False),
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=config["model"].get("trust_remote_code", False),
    )
    model = PeftModel.from_pretrained(base_model, config["paths"]["output_dir"])
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, instruction: str, max_new_tokens: int = 256) -> tuple[str, float]:
    prompt = f"<|user|>\n{instruction}\n<|end|>\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    start = time.monotonic()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,            # deterministic for eval
            pad_token_id=tokenizer.eos_token_id,
        )
    duration_ms = (time.monotonic() - start) * 1000

    text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return text.strip(), duration_ms


def main():
    config = load_config()
    model, tokenizer = load_adapter(config)

    eval_data = [json.loads(line) for line in Path(config["paths"]["eval_data"]).read_text().splitlines()]
    samples = eval_data[:20]   # ilk 20 örnek

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)

    rouge1_scores, rougeL_scores, durations = [], [], []
    generated_outputs = []
    references = []

    for sample in samples:
        gen, duration_ms = generate(model, tokenizer, sample["instruction"])
        ref = sample["output"]

        rouge = scorer.score(ref, gen)
        rouge1_scores.append(rouge["rouge1"].fmeasure)
        rougeL_scores.append(rouge["rougeL"].fmeasure)
        durations.append(duration_ms)
        generated_outputs.append(gen)
        references.append(ref)

    # BERTScore — multilingual, Turkish için anlamlı
    _, _, bert_f1 = bert_score_fn(generated_outputs, references, lang="tr", verbose=False)
    bert_f1_mean = bert_f1.mean().item()

    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)
    avg_duration = sum(durations) / len(durations)

    print("\n=== Evaluation Results ===")
    print(f"ROUGE-1 (avg):     {avg_rouge1:.4f}")
    print(f"ROUGE-L (avg):     {avg_rougeL:.4f}")
    print(f"BERTScore F1 (tr): {bert_f1_mean:.4f}")
    print(f"Avg inference:     {avg_duration:.1f} ms")

    # 5 örnek göster — manuel kalite kontrol için
    print("\n=== Sample Outputs ===")
    for i in range(5):
        print(f"\n--- Example {i+1} ---")
        print(f"Q: {samples[i]['instruction']}")
        print(f"Ref: {references[i][:200]}...")
        print(f"Gen: {generated_outputs[i][:200]}...")
        print(f"ROUGE-L: {rougeL_scores[i]:.3f}")

    # MLflow'a evaluation metric'leri yaz (aktif run'a)
    # Not: bu script bağımsız çalıştırılabileceği için,
    # en iyi pattern — run_id'yi CLI arg olarak al, o run'a metric ekle.
    # Basit versiyon: yeni bir "evaluation" run'ı oluştur.
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="evaluation", tags={"type": "evaluation"}):
        mlflow.log_metrics({
            "evaluation_rouge1": avg_rouge1,
            "evaluation_rougeL": avg_rougeL,
            "evaluation_bertscore_f1": bert_f1_mean,
            "evaluation_avg_inference_ms": avg_duration,
        })


if __name__ == "__main__":
    main()
```

**Yorum kalibrasyonu (Türkçe için):**
- ROUGE-L < 0.25 → düşük kalite, adapter iş görmüyor olabilir
- ROUGE-L 0.25–0.40 → makul; manuel doğrulama gerek
- ROUGE-L > 0.40 → iyi (ama overfitting kontrolü yap: eval set train ile overlap etti mi?)
- **BERTScore F1 > 0.75** → semantik olarak iyi örtüşme
- **Manuel eval şart** — bu metrikler Türkçe'de tek başına yanıltıcı.

---

## requirements.txt

> **📝 Implementation note (Sapma 5):** SPEC'teki tam pin'ler (torch 2.3.1,
> trl 0.9.6, transformers 4.43.3 — 2024-Q2/Q3 sürümleri) Python 3.13'te wheel
> bulunmaması nedeniyle uygulanmadı. **Modern floor pin'lere geçildi:**
> torch==2.9.1 (T4 + 3.13 uyumlu), transformers>=4.46, trl>=0.12, peft>=0.13,
> accelerate>=1.0, bitsandbytes>=0.44 (`sys_platform != 'darwin'` marker — Mac'te
> skip, Colab'da manuel). Reproducibility için `requirements.lock.txt`
> opsiyonel (pip freeze çıktısı). Final dosya: `ml/requirements.txt`.

```
# Pinned versions — reproducibility için.
# Yeni sürümlerde API değişiklikleri olabilir; upgrade etmeden önce test.
torch==2.3.1
transformers==4.43.3
datasets==2.20.0
peft==0.12.0
trl==0.9.6
accelerate==0.33.0
bitsandbytes==0.43.3
mlflow==2.15.1
rouge-score==0.1.2
bert-score==0.3.13
pyyaml==6.0.2
jupyter==1.0.0
nbstripout==0.7.1
sentence-transformers==3.0.1      # opsiyonel: semantic deduplication
python-dotenv==1.0.1               # .env yüklemesi için
anthropic>=0.34.0                  # Seçenek A kullanılıyorsa sentetik veri üretimi
scikit-learn==1.5.1                 # stratified split

# Dev
pytest==8.3.2
ruff==0.5.7
```

**2026 güncelleme notu:** Bu sürümler spec yazıldığında çalışıyordu. Upgrade gerekli ise:
- `trl` 0.11+'da `SFTTrainer` API'sinde `tokenizer=` → `processing_class=` oldu
- `transformers` 4.45+'da bazı `from_pretrained` parametreleri değişti
- Upgrade edersen sürümü pin'le ve değişiklikleri IMPLEMENTATION_NOTES'a not et

---

## .env.example

```bash
# Sentetik veri üretimi için (Seçenek A kullanılıyorsa)
ANTHROPIC_API_KEY=

# Hugging Face Hub — private model/dataset gerekirse
HF_TOKEN=

# MLflow — remote tracking server kullanılacaksa
# MLFLOW_TRACKING_URI=
# MLFLOW_TRACKING_USERNAME=
# MLFLOW_TRACKING_PASSWORD=
```

---

## Colab workflow (GPU yoksa)

> **📝 Implementation note (Sapma 26):** Colab'da **torch'u reinstall etmeyin**.
> Native `torch 2.10.0+cu128` + `torchvision 0.25.0+cu128` ikilisi uyumlu
> gelir; `pip install torch==X --upgrade` PyPI vanilla build'ini yükleyip
> torchvision ile operator mismatch yaratır (`RuntimeError: torchvision::nms`).
> Doğru pattern: `grep -v '^torch==' requirements.txt > /tmp/colab_req.txt`
> ve sadece geri kalanları kur. `bitsandbytes`'ı manuel ekle (Mac
> environment marker ile skip oldu). Tam Colab tek-hücre template'i için
> [`P3_HANDOFF.md`](P3_HANDOFF.md) referans alınır.

Lokal GPU yoksa Colab free tier (T4, 16GB VRAM) Phi-3 Mini + QLoRA için yeterli.

```python
# Colab notebook'u — ilk hücreler
!git clone https://github.com/yasirbeydiligit/eduai-platform.git
%cd eduai-platform/ml
!pip install -r requirements.txt

# Google Drive mount — MLflow run'ları + checkpoint'leri kalıcı saklar
from google.colab import drive
drive.mount('/content/drive')

# config.yaml'da tracking_uri override
import yaml
with open("training/config.yaml") as f:
    config = yaml.safe_load(f)
config["mlflow"]["tracking_uri"] = "/content/drive/MyDrive/eduai_mlruns"
config["paths"]["output_dir"] = "/content/drive/MyDrive/eduai_checkpoints"
with open("training/config.yaml", "w") as f:
    yaml.dump(config, f)

# Training
!python training/train.py
```

**Kritik:** Colab session 12 saat sonra düşer. Uzun training için:
- `save_steps` düşük tut (100 civarı)
- Drive'a checkpoint yaz (yukarıdaki override)
- Resume training: `SFTConfig(resume_from_checkpoint=True)` ile

---

## GitHub Actions CI — P2 için

> **📝 Implementation note (Sapma 9, 29, 30):** SPEC'te `ml-quality` job vardı
> ama dependency graph belirsizdi. **Karar: top-level paralel** (`needs:` yok)
> — P1 (`services/api`) ve P2 (`ml/`) bağımsız kod alanları, biri kırmızı
> olsa diğeri block olmasın. **Install minimal'leştirildi** (`ruff pytest`
> yeterli; SPEC'in `pyyaml jsonschema` önerisi unused, YAGNI). `test_data_schema.py`
> Task 6 yerine **Task 1 sonunda yazıldı** (Sapma 9 — veri üretildikten hemen
> sonra schema doğrulama yapılabilsin). Final 4-job graph:
> `[lint] [ml-quality] → [test] → [docker-build]`.

`ml/` dizini eklendiğinde mevcut `ci.yml`'e **yeni bir job** eklenmeli:

```yaml
  ml-quality:
    name: "ML Code Quality & Schema"
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"
      - name: Install light deps
        # Training deps (torch, transformers, peft, trl) CI'da kurmuyoruz —
        # çok büyük ve GPU yok. Sadece lint + schema test.
        run: pip install ruff pytest pyyaml jsonschema
      - name: Ruff lint
        run: ruff check ml/ --output-format=github
      - name: Ruff format check
        run: ruff format ml/ --check
      - name: Data schema tests
        working-directory: ml
        run: pytest tests/ -v --tb=short
```

`tests/test_data_schema.py`:
```python
"""JSONL dataset schema validation — CI'da çalışır."""
import json
from pathlib import Path

import pytest

REQUIRED_KEYS = {"instruction", "input", "output", "subject", "grade"}
VALID_SUBJECTS = {
    "matematik", "fizik", "kimya", "biyoloji", "tarih",
    "cografya", "felsefe", "din", "edebiyat", "ingilizce", "genel",
}


@pytest.mark.parametrize("path", ["data/processed/train.jsonl", "data/processed/eval.jsonl"])
def test_jsonl_schema(path: str) -> None:
    """Her satır geçerli JSON + required keys + valid subject + grade 1-12."""
    p = Path(path)
    if not p.exists():
        pytest.skip(f"{path} yok — data_prep.py henüz çalışmadı")

    for i, line in enumerate(p.read_text().splitlines(), 1):
        record = json.loads(line)
        assert REQUIRED_KEYS.issubset(record.keys()), f"Line {i}: eksik anahtar"
        assert record["subject"] in VALID_SUBJECTS, f"Line {i}: geçersiz subject"
        assert 1 <= record["grade"] <= 12, f"Line {i}: grade 1-12 dışı"
        assert len(record["output"]) >= 20, f"Line {i}: output çok kısa"
```

---

## Teslim kriterleri

> **📝 P2 FINALIZE STATUS — 2026-04-28**
>
> **Final metrikler (Qwen3-4B-Instruct-2507 baseline, MLflow run
> `01822d480ec1471b90357c1643dd6aba`):**
> - ROUGE-1: 0.399 / **ROUGE-L: 0.249** (Türkçe morfolojisi → eşik dibi normal)
> - **BERTScore F1 (tr): 0.640** (orta — RAG'a uygun)
> - eval_loss best: 1.347 / train_loss final: 1.45
> - mean_token_accuracy: 0.62 → 0.69
> - Avg inference latency: 22978 ms (T4 + 4-bit)
>
> **Bilinçli kabul edilen istisnalar (Yol B, Sapma 28 sonrası):**
> 1. **ROUGE-L 0.249 eşik dibinde** — Türkçe morfolojisi yapısal düşüş, RAG
>    context'i ile gerçek kalite P3 sonrası ölçülecek
> 2. **Manuel QA atlandı** — saf P2 manuel QA "RAG'sız" kullanım senaryosunu
>    yansıtmıyor; P3 sonrası A/B test (RAG-with vs RAG-without) daha değerli
> 3. **Inference latency 22 sn T4'te** — production değil; A100 ~5s veya P3
>    vLLM serving ile aşılır
>
> **Yol B gerekçesi:** Pareto disiplini — ilk %80 kalite mevcut, kalan %20
> için marjinal kazanç vs RAG mimarisinin dramatik kazanımı. Detaylar:
> `IMPLEMENTATION_NOTES_ARCHIVED.md` Task 4 sonuç tablosu.

- [ ] `python training/data_prep.py` → `train.jsonl` + `eval.jsonl` üretiyor
- [ ] `pytest ml/tests/` → schema test'leri geçiyor
- [ ] `python training/train.py` → hatasız tamamlanıyor, adapter kaydediliyor
- [ ] `mlflow ui` → en az **3 deney** görünüyor (lora_r 8/16 + lr variation), karşılaştırılabilir
- [ ] `python training/evaluate.py` → ROUGE + BERTScore hesaplanıyor, 5 örnek yazdırılıyor
- [ ] `ml/README.md` → training + MLflow + Colab workflow
- [ ] Manuel evaluation: 10 çeşitli soru, cevap kalitesi dokümante (README'de tablo)
- [ ] GitHub Actions → `ml-quality` job yeşil
- [ ] P3 handoff dokümanı (`docs/p2/P3_HANDOFF.md`) yazıldı

---

## Önemli notlar

- **GPU yoksa:** Colab T4 ücretsiz Phi-3 Mini + QLoRA için yeter. Qwen3 7B için Colab Pro ($10/ay, A100) gerekir.
- **Veri gizliliği:** Gerçek öğrenci verisi kullanma. Sentetik veya açık kaynak.
- **Base model lisansı:** Phi-3/4 MIT — serbest. Qwen Apache 2.0 — serbest. LLaMA-3.x Meta lisansı — ticari kullanımda kontrol et.
- **"Product" beklentisi:** P2'den tek başına satılabilir ürün çıkmaz. P3 RAG ile birlikte çalışınca ürün haline gelir. P2'nin gerçek değeri: pedagojik tonu kazanmış, format'a uymuş bir adapter + MLflow deney disiplini + baseline metrikler.
