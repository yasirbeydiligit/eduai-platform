# P2 — Model Fabrikası: Proje Spesifikasyonu

---

## Proje özeti

**Ne:** Türkçe lise eğitimi için Phi-3 Mini (veya Qwen2 7B) modelini QLoRA ile fine-tune eden, MLflow ile deney takibi yapan ML pipeline.
**Klasör:** `eduai-platform/ml/`
**Bağımlılık:** P1'e dokunmaz. Bağımsız çalışır, çıktısı P3'te kullanılır.

---

## Klasör yapısı

```
eduai-platform/
└── ml/
    ├── data/
    │   ├── raw/
    │   │   └── eduai_qa_raw.jsonl          ← ham Q&A verisi
    │   └── processed/
    │       ├── train.jsonl                  ← eğitim seti (%80)
    │       └── eval.jsonl                   ← değerlendirme seti (%20)
    ├── training/
    │   ├── config.yaml                      ← tüm hyperparametreler burada
    │   ├── train.py                         ← ana eğitim scripti
    │   ├── evaluate.py                      ← ROUGE + perplexity
    │   └── data_prep.py                     ← ham veriyi işler
    ├── models/
    │   └── checkpoints/                     ← LoRA adaptörleri
    ├── notebooks/
    │   └── 01_data_exploration.ipynb        ← buradan başla
    ├── requirements.txt
    └── README.md
```

---

## Dataset spesifikasyonu

### Format: JSONL (her satır bir JSON)

```jsonl
{"instruction": "Türkiye'de İkinci Meşrutiyet'in ilan edilmesinin sebeplerini açıkla.", "input": "", "output": "İkinci Meşrutiyet, 1908 yılında İttihat ve Terakki Cemiyeti'nin baskısıyla ilan edilmiştir. Temel sebepler: ordunun siyasallaşması, II. Abdülhamit'in istibdat yönetimine duyulan tepki, Makedonya sorunu ve Batılı devletlerin Osmanlı topraklarına yönelik müdahalelerdir.", "subject": "tarih", "grade": 10}
{"instruction": "İkinci dereceden bir denklemi çöz: 2x² + 5x - 3 = 0", "input": "", "output": "Discriminant: b² - 4ac = 25 + 24 = 49. Kökler: x = (-5 ± 7) / 4. Sonuç: x₁ = 0.5, x₂ = -3", "subject": "matematik", "grade": 9}
```

### Gerekli örnek sayısı
- Minimum: 200 örnek (hızlı deney için)
- Hedef: 500 örnek (iyi kalite için)
- Dağılım: her ders için en az 50 örnek

### Veri oluşturma stratejisi
Veri yoksa Claude API ile sentetik veri üret:
```python
# data_prep.py içinde
# Claude API'ye toplu soru-cevap ürettir
# Sonra manuel kalite kontrolü yap
```

---

## config.yaml

```yaml
model:
  name: "microsoft/Phi-3-mini-4k-instruct"  # veya "Qwen/Qwen2-7B-Instruct"
  max_length: 512

lora:
  r: 16                    # LoRA rank (8-64 arası dene)
  lora_alpha: 32           # genellikle r * 2
  lora_dropout: 0.05
  target_modules:          # hangi katmanlar fine-tune edilecek
    - "q_proj"
    - "v_proj"
    - "k_proj"
    - "o_proj"

quantization:
  load_in_4bit: true       # QLoRA için
  bnb_4bit_compute_dtype: "float16"
  bnb_4bit_quant_type: "nf4"

training:
  num_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4  # effective batch = 16
  learning_rate: 2e-4
  warmup_ratio: 0.03
  lr_scheduler_type: "cosine"
  save_steps: 100
  eval_steps: 100
  logging_steps: 10

paths:
  train_data: "data/processed/train.jsonl"
  eval_data: "data/processed/eval.jsonl"
  output_dir: "models/checkpoints"
  mlflow_tracking_uri: "mlruns"

mlflow:
  experiment_name: "eduai-fine-tuning"
  run_tags:
    project: "EduAI Platform P2"
```

---

## train.py — tam mimari

```python
"""
EduAI Fine-tuning Pipeline
Phi-3 Mini → QLoRA → Türkçe eğitim Q&A
"""
import mlflow
import yaml
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
import torch

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def format_prompt(example: dict) -> str:
    """Instruction → model input formatına çevir"""
    return f"""<|user|>
{example['instruction']}
<|end|>
<|assistant|>
{example['output']}
<|end|>"""

def setup_model(config: dict):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config["quantization"]["load_in_4bit"],
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type=config["quantization"]["bnb_4bit_quant_type"],
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"],
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def apply_lora(model, config: dict):
    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        lora_dropout=config["lora"]["lora_dropout"],
        target_modules=config["lora"]["target_modules"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    return get_peft_model(model, lora_config)

def main():
    config = load_config("training/config.yaml")
    
    mlflow.set_tracking_uri(config["mlflow"]["mlflow_tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    
    with mlflow.start_run(tags=config["mlflow"]["run_tags"]):
        # Hyperparametreleri kaydet
        mlflow.log_params({
            "model": config["model"]["name"],
            "lora_r": config["lora"]["r"],
            "learning_rate": config["training"]["learning_rate"],
            "epochs": config["training"]["num_epochs"],
        })
        
        # Model kur
        model, tokenizer = setup_model(config)
        model = apply_lora(model, config)
        model.print_trainable_parameters()  # kaç parametre eğitilecek?
        
        # Veri yükle
        dataset = load_dataset("json", data_files={
            "train": config["paths"]["train_data"],
            "eval": config["paths"]["eval_data"],
        })
        
        # Trainer
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["eval"],
            formatting_func=format_prompt,
            args=SFTConfig(
                output_dir=config["paths"]["output_dir"],
                num_train_epochs=config["training"]["num_epochs"],
                per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
                learning_rate=config["training"]["learning_rate"],
                logging_steps=config["training"]["logging_steps"],
                eval_steps=config["training"]["eval_steps"],
                save_steps=config["training"]["save_steps"],
                max_seq_length=config["model"]["max_length"],
            ),
        )
        
        # Eğit
        train_result = trainer.train()
        
        # Metrikleri MLflow'a yaz
        mlflow.log_metrics({
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics["train_runtime"],
        })
        
        # Adaptörü kaydet
        trainer.model.save_pretrained(config["paths"]["output_dir"])
        mlflow.log_artifacts(config["paths"]["output_dir"])
        
        print(f"Training complete. Run ID: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    main()
```

---

## evaluate.py gereksinimleri

```python
# Şunları hesapla ve MLflow'a yaz:
# 1. ROUGE-1, ROUGE-L skoru (rouge-score kütüphanesi)
# 2. 10 örnek soruya model cevabı + referans cevap yan yana yazdır
# 3. Ortalama inference süresi (ms)
```

---

## requirements.txt

```
torch>=2.1.0
transformers>=4.40.0
datasets>=2.18.0
peft>=0.10.0
trl>=0.8.0
bitsandbytes>=0.43.0
accelerate>=0.28.0
mlflow>=2.12.0
rouge-score>=0.1.2
pyyaml>=6.0
jupyter>=1.0.0
```

---

## Teslim kriterleri

- [ ] `python training/train.py` çalışıyor, hata vermeden tamamlanıyor
- [ ] `mlflow ui` → experiment kaydedilmiş, metrikler görünüyor
- [ ] En az 2 farklı `lora_r` değeriyle (8 ve 16) deney yapıldı
- [ ] `evaluate.py` → ROUGE skoru hesaplandı
- [ ] LoRA adaptörü `models/checkpoints/` altına kaydedildi
- [ ] README: eğitim nasıl çalıştırılır + MLflow nasıl başlatılır

---

## Önemli notlar

**GPU yoksa:** Google Colab free tier (T4 GPU) Phi-3 Mini + QLoRA için yeterli.
**Colab'da MLflow:** `mlflow ui --host 0.0.0.0` komutu + ngrok tunnel kullanılabilir, ya da basit CSV loglama yapılabilir.
**Veri gizliliği:** Gerçek öğrenci verisi kullanma. Sentetik veri üret veya açık kaynak Türkçe eğitim dataseti kullan.
