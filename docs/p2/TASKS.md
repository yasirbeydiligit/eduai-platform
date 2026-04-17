# P2 — Görev Listesi: Model Fabrikası

> P1'den bağımsız ilerler. Paralel başlayabilirsin.
> Her task sonunda `git commit` at.

---

## Task 0 — Ortam kur ⏱ ~20 dk

### Claude Code'a ver:
```
Proje: eduai-platform/ml/

Görev: ML proje yapısını oluştur.

Klasörler:
ml/
├── data/raw/
├── data/processed/
├── training/
├── models/checkpoints/
├── notebooks/
└── requirements.txt (SPEC.md'deki paketleri ekle)

Ayrıca:
- .gitignore: models/checkpoints/ klasörünü ekle (büyük dosyalar)
- data/.gitignore: raw veri de ignore (gizlilik)
- notebooks/01_data_exploration.ipynb: boş notebook oluştur, başlık hücresi ekle
```

### Bunu kendin yap:
```bash
pip install -r ml/requirements.txt
git add . && git commit -m "feat: P2 ML project structure"
```

---

## Task 1 — Veri seti oluştur ⏱ ~1 saat

**Bu task'ta öğreniyorsun:** JSONL formatı, prompt engineering, veri kalitesi.

### Claude Code'a ver:
```
Proje: eduai-platform/ml/

Görev: training/data_prep.py yaz.

Amacı: Türkçe lise Q&A dataset'i oluşturup JSONL formatında kaydet.

1. Sentetik veri üretimi için örnek veri listesi:
   Aşağıdaki konular için her birinden 10 soru-cevap çifti üret:
   - Tarih: Osmanlı dönemi, Kurtuluş Savaşı, Cumhuriyet tarihi
   - Matematik: 2. dereceden denklemler, türev, integral temelleri
   - Fizik: Newton yasaları, elektrik devreleri, optik
   - Kimya: Asit-baz, mol hesabı, periyodik tablo
   - Edebiyat: Türk şiiri dönemleri, roman analizi

2. Her kayıt şu formatta olsun:
   {"instruction": "...", "input": "", "output": "...", "subject": "tarih", "grade": 9}

3. output: en az 3 cümle, pedagojik ton, 9. sınıf seviyesi

4. %80 train / %20 eval split yap:
   - data/processed/train.jsonl
   - data/processed/eval.jsonl

5. Veri istatistiklerini yazdır:
   - Toplam örnek sayısı
   - Her ders için örnek sayısı
   - Ortalama output uzunluğu (token)
```

### Bunu kendin yap:
```bash
python ml/training/data_prep.py
# Kaç örnek üretildi? train/eval dağılımı doğru mu?
# data/processed/ klasörüne bak
git add . && git commit -m "feat: generate training dataset for P2"
```

---

## Task 2 — config.yaml ve train.py ⏱ ~30 dk

### Claude Code'a ver:
```
Proje: eduai-platform/ml/

Görev: training/config.yaml ve training/train.py yaz.
SPEC.md'deki tam kodu kullan. Değişiklik yapma, birebir uygula.

Ekstra gereksinim:
- train.py başında model'in trainable parameter sayısını yazdır
- Her 10 adımda konsola "Step X | Loss: Y" yazdır
- Training bitince "Training complete!" + toplam süre yazdır
```

### Bunu kendin yap:
```bash
# Önce küçük test: 10 step ile dene
# config.yaml'da: num_epochs: 1, max_steps: 10 ekle
python ml/training/train.py

# Hata aldın mı? En yaygın hatalar:
# "CUDA out of memory" → batch_size'ı 2'ye düşür
# "Module not found" → pip install eksik paket
# "No such file" → data/processed/ klasörü var mı?

git add . && git commit -m "feat: training pipeline with LoRA + MLflow"
```

---

## Task 3 — MLflow görselleştirme ⏱ ~20 dk

**Bu task'ta öğreniyorsun:** Experiment tracking, metrik karşılaştırma.

### Claude Code'a ver:
```
Proje: eduai-platform/ml/

Görev: İki farklı LoRA konfigürasyonu ile deney yap.

1. config.yaml'da lora.r = 8 ile bir run başlat:
   run adı: "lora-r8-lr2e4"

2. config.yaml'da lora.r = 16 ile bir run başlat:
   run adı: "lora-r16-lr2e4"

3. Her run için MLflow'a şunları logla:
   - lora_r, learning_rate, num_epochs (param)
   - train_loss (her 10 step'te metric)
   - total_trainable_params (metric)
   - training_duration_seconds (metric)

mlflow ui'da iki run yan yana karşılaştırılabilmeli.
```

### Bunu kendin yap:
```bash
mlflow ui
# Tarayıcı: http://localhost:5000
# İki run'ı seç → Compare
# Hangi lora_r daha iyi loss verdi?

git add . && git commit -m "exp: compare lora-r8 vs lora-r16 configurations"
```

---

## Task 4 — Evaluation ⏱ ~30 dk

**Bu task'ta öğreniyorsun:** NLP metrikleri, model kalitesi ölçümü.

### Claude Code'a ver:
```
Proje: eduai-platform/ml/

Görev: training/evaluate.py yaz.

1. En iyi checkpoint'i yükle (models/checkpoints/)
2. eval.jsonl'den ilk 20 örneği al
3. Her örnek için:
   - Model cevabı üret (max 256 token)
   - ROUGE-1 ve ROUGE-L hesapla
   - inference süresini ölç (ms)
4. Sonuçları yazdır:
   - Ortalama ROUGE-1 skoru
   - Ortalama ROUGE-L skoru  
   - Ortalama inference süresi
5. 5 örnek için yan yana yazdır:
   - Soru
   - Referans cevap
   - Model cevabı
   - ROUGE-L skoru

Sonuçları MLflow'a da yaz (evaluation_ prefix ile).
```

### Bunu kendin yap:
```bash
python ml/training/evaluate.py
# ROUGE-L > 0.3 çıktıysa iyi başlangıç
# Cevaplar okunabilir mi? Türkçe doğru mu?

git add . && git commit -m "eval: add ROUGE evaluation for fine-tuned model"
```

---

## Task 5 — Notebook tamamla ⏱ ~30 dk

### Claude Code'a ver:
```
Proje: eduai-platform/ml/

Görev: notebooks/01_data_exploration.ipynb'i tamamla.

Notebook şunları içermeli (her biri ayrı hücre):

Hücre 1: Başlık + açıklama (Markdown)
Hücre 2: Veri yükleme + ilk 5 örneği göster
Hücre 3: İstatistikler (toplam, ders dağılımı, uzunluklar)
Hücre 4: Bar chart - ders başına örnek sayısı (matplotlib)
Hücre 5: Histogram - output token uzunlukları
Hücre 6: Tokenizer analizi - en uzun/en kısa örnekler
Hücre 7: Sonuç ve notlar (Markdown)
```

### Bunu kendin yap:
```bash
jupyter notebook ml/notebooks/01_data_exploration.ipynb
# Tüm hücreleri çalıştır, çıktılar görünüyor mu?

git add . && git commit -m "docs: add data exploration notebook"
```

---

## P2 tamamlandı mı? Kontrol listesi

```
[ ] data/processed/train.jsonl ve eval.jsonl var (min 160 örnek)
[ ] python training/train.py → hatasız çalışıyor
[ ] models/checkpoints/ → LoRA adaptörü kaydedildi
[ ] mlflow ui → 2 deney görünüyor, karşılaştırılabiliyor
[ ] evaluate.py → ROUGE skoru hesaplanıyor
[ ] Notebook tüm çıktılarıyla kaydedilmiş
[ ] README: training nasıl çalıştırılır, MLflow nasıl başlatılır
```

P3'e geçmeden önce: modelin eval.jsonl'de makul cevaplar üretip üretmediğini manuel kontrol et. Tamamen anlamsız çıktılar varsa training'e devam et veya learning_rate'i ayarla.
