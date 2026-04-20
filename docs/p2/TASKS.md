# P2 — Görev Listesi: Model Fabrikası

> P1'den bağımsız ilerler (paralel başlayabilir). Her task sonunda `git commit` at.
> **P1 prensibi geçerli:** SPEC'teki kodu körü körüne uygulama. Sürüm uyumsuzluğu/mantıksız nokta varsa **sebebiyle birlikte düzelt** ve `docs/p2/IMPLEMENTATION_NOTES.md`'de not et.

---

## Task 0 — Ortam kur ⏱ ~30 dk

### Claude Code'a ver:
```
Proje: eduai-platform/ml/

Görev: ML proje yapısını oluştur (SPEC.md'deki klasör yapısı).

Klasörler:
ml/
├── data/raw/              (.gitignore'a ekle — gizlilik)
├── data/processed/
├── training/
├── models/checkpoints/    (.gitignore'a ekle — büyük dosyalar)
├── notebooks/
├── tests/                 (SPEC.md'de test_data_schema.py yazılacak yer)
├── requirements.txt       (SPEC'teki PINNED sürümleri birebir kopyala)
├── .env.example           (SPEC'teki örnek)
└── README.md              (şimdilik başlık + 1-2 satır; Task 7'de genişletilecek)

Kök-seviye .gitignore güncellemeleri:
- ml/models/checkpoints/
- ml/data/raw/
- ml/mlruns/              (MLflow local tracking)
- ml/.env                  (credential'lar)
- notebooks/.ipynb_checkpoints/

Ayrıca:
- nbstripout kur: `nbstripout --install` (notebook output'ları git diff'lerini kirletir)
- notebooks/01_data_exploration.ipynb: boş notebook oluştur, başlık + amaç hücresi ekle
```

### Bunu kendin yap:
```bash
cd eduai-platform
pip install -r ml/requirements.txt    # lokal test — bazı paket GPU'ya ihtiyaç duymaz
nbstripout --install                   # git output-stripping
git add . && git commit -m "feat(p2): ML project structure with pinned deps"
```

---

## Task 1 — Veri seti oluştur ⏱ ~2-3 saat

**Bu task'ta öğreniyorsun:** JSONL formatı, prompt engineering, veri kalitesi, stratified split.

> ⚠️ **Zaman tahmini realistic:** SPEC'teki "1 saat" yanıltıcı. Seçenek A (Claude API) için: template hazırlama (30 dk) + toplu üretim (30-60 dk API calls) + manuel QA pass (1-2 saat) = toplam 2-3 saat.

### Claude Code'a ver:
```
Proje: eduai-platform/ml/

Görev: training/data_prep.py yaz — Türkçe lise Q&A dataset'i üretip JSONL'e kaydet.

ÖNCE STRATEJI SEÇ (SPEC.md Dataset bölümü):
- Seçenek A (Sentetik, Claude API): hızlı, ölçeklenir, QA gerek
- Seçenek B (Açık kaynak dönüştürme): yavaş ama orijinal
- Seçenek C (Hibrit): önerilen

Seçimini ml/README.md'de dokümante et.

data_prep.py requirements:

1. Seçenek A ise:
   - .env'den ANTHROPIC_API_KEY oku (python-dotenv)
   - Her subject için prompt template:
     "Sen 9-12. sınıf öğrencileri için Türkçe eğitim asistanısın.
      {subject} konusunda {grade}. sınıf seviyesine uygun N adet
      soru-cevap çifti üret. Format: JSON list..."
   - Subject'ler: tarih, matematik, fizik, kimya, biyoloji,
                  cografya, felsefe, edebiyat, ingilizce
   - Hedef: her subject için 50 örnek → toplam ~450

2. Her kayıt format:
   {"instruction": "...", "input": "",
    "output": "en az 3 cümle, pedagojik ton",
    "subject": "tarih", "grade": 9}

3. Kalite filtreleri:
   - output < 20 karakter → ele
   - output > 1000 karakter → ele
   - Exact duplicate instruction → ele
   - (opsiyonel) Semantic dedup: sentence-transformers cosine > 0.95

4. Stratified %80/%20 split (sklearn.train_test_split, stratify=subject+grade):
   - data/processed/train.jsonl
   - data/processed/eval.jsonl

5. İstatistikler yazdır:
   - Toplam örnek sayısı
   - Her subject × grade için dağılım
   - Ortalama output token uzunluğu (tokenizer'la)
   - Kaç örnek quality filter'a takıldı

6. Random seed (42) → reproducibility

Manuel QA checklist (README'de belirt):
- 20 rastgele örneği göz at
- Pedagojik ton uygun mu?
- Grade-level'e göre karmaşıklık doğru mu?
- Türkçe dilbilgisi doğru mu?
```

### Bunu kendin yap:
```bash
cp ml/.env.example ml/.env
# ml/.env içine ANTHROPIC_API_KEY ekle (seçenek A)

python ml/training/data_prep.py
# Kaç örnek üretildi? Dağılım dengeli mi?

# Manuel QA: rastgele 20 örneği aç
head -20 ml/data/processed/train.jsonl | python -m json.tool

# Schema test'i çalıştır
cd ml && pytest tests/ -v
cd ..

git add ml/ && git commit -m "feat(p2): generate training dataset with quality filters"
```

---

## Task 2 — config.yaml ve train.py ⏱ ~1-1.5 saat

### Claude Code'a ver:
```
Proje: eduai-platform/ml/

Görev: training/config.yaml ve training/train.py yaz.

REFERANS: SPEC.md'deki config.yaml ve train.py mimari yol haritasıdır.
SÜRÜM UYUMSUZLUĞU KONTROLÜ:
  pip show trl transformers peft
  trl >= 0.11 → SFTTrainer(processing_class=tokenizer)
  trl < 0.11 → SFTTrainer(tokenizer=tokenizer)
  SFTConfig'in tüm parametrelerini `trl.trainer.sft_config` docstring'inde doğrula.

Uyuşmayan API varsa:
  1. Düzelt (çalışır hale getir)
  2. docs/p2/IMPLEMENTATION_NOTES.md'ye not ekle (sebep + dosya:satır)

Ekstra gereksinimler:
- set_seed(config["seed"]) mutlaka train başında
- Trainable parameter count MLflow'a metric olarak yaz
- Her N adımda konsola "Step X | Loss: Y | LR: Z" yazdır (N=logging_steps)
- Training bitince süre + MLflow run_id yazdır
- Keyboard interrupt (Ctrl+C) → trainer.state kaydet, adapter save, graceful exit

Smoke test modu (opsiyonel CLI arg):
- python train.py --smoke → max_steps=10, num_epochs=1 (hızlı pipeline doğrulama)
- Full training için argüman yok
```

### Bunu kendin yap:
```bash
# Önce smoke test — pipeline çalışıyor mu?
python ml/training/train.py --smoke
# Adapter oluştu mu? MLflow'a yazdı mı?

# Full training
python ml/training/train.py

# Yaygın hatalar:
# "CUDA out of memory" → per_device_train_batch_size: 2, grad_accum: 8
# "Module not found" → pip install eksik paket
# "No such file" → data/processed/ klasörü var mı?
# "Trust remote code" prompt → config.yaml'da trust_remote_code: true
# TRL AttributeError → sürüm farkı, SPEC uyarısına dön, uyarla

git add . && git commit -m "feat(p2): training pipeline — QLoRA + MLflow + seed"
```

---

## Task 3 — Deney karşılaştırması ⏱ ~1-2 saat

**Bu task'ta öğreniyorsun:** Hyperparameter sweep, MLflow comparison, overfitting tespiti.

### Claude Code'a ver:
```
Proje: eduai-platform/ml/

Görev: 3-4 farklı konfigürasyonla deney yap, MLflow'da karşılaştır.

Denemeler (config.yaml'ı override ederek veya 4 farklı config dosyasıyla):

1. run_name: "lora-r8-lr2e4"     → lora.r=8, lr=2e-4
2. run_name: "lora-r16-lr2e4"    → lora.r=16, lr=2e-4
3. run_name: "lora-r16-lr5e5"    → lora.r=16, lr=5e-5
4. run_name: "lora-r32-lr2e4"    → lora.r=32, lr=2e-4 (opsiyonel, VRAM yeterse)

Her run için MLflow'a logla (SPEC'teki parametrelere ek olarak):
- lora_r, lora_alpha, learning_rate (param)
- train_loss, eval_loss (her eval_steps'te metric)
- total_trainable_params (metric)
- training_duration_seconds (metric)
- grad_norm (opsiyonel, overfitting sinyali)

Analiz (bu run'lar bittikten sonra):
- mlflow ui --port 5000 → http://localhost:5000
- Experiments > eduai-fine-tuning > compare
- Plot: train_loss vs eval_loss (tüm run'lar)
- Overfitting var mı? (eval_loss düşüşü duruyor ama train_loss düşmeye devam ediyor mu?)
- Hangi (r, lr) kombinasyonu en düşük eval_loss verdi?

Bulguları docs/p2/IMPLEMENTATION_NOTES.md'ye yaz (tablo: run_name, lora_r, lr, final_eval_loss, duration).
```

### Bunu kendin yap:
```bash
# 4 run'ı sırayla veya paralel (Colab sessizce reset'lenir, lokal daha güvenli)
python ml/training/train.py   # her seferinde config'i değiştir

mlflow ui --port 5000
# Karşılaştırma: plot > line chart > train_loss, eval_loss

git add . && git commit -m "exp(p2): lora rank + lr hyperparameter sweep"
```

---

## Task 4 — Evaluation ⏱ ~1-1.5 saat

**Bu task'ta öğreniyorsun:** NLP metrikleri, Türkçe için metric kalibrasyonu, manuel QA.

### Claude Code'a ver:
```
Proje: eduai-platform/ml/

Görev: training/evaluate.py yaz — SPEC.md'deki tam implementation baz alınır.

Adım adım:
1. En iyi checkpoint'i yükle (models/checkpoints/ — PeftModel.from_pretrained)
2. eval.jsonl'den ilk 20 örnek al
3. Her örnek için model.generate:
   - do_sample=False (deterministic)
   - max_new_tokens=256
   - inference süresi ölç (ms)
4. Metric hesapla:
   - ROUGE-1, ROUGE-L (rouge-score)
   - BERTScore F1 (lang="tr", bert-score paketi)
5. 5 örnek için yan yana yazdır:
   - Soru | Referans | Üretilen | ROUGE-L
6. Ortalamaları MLflow'a logla (evaluation_ prefix):
   - evaluation_rouge1
   - evaluation_rougeL
   - evaluation_bertscore_f1
   - evaluation_avg_inference_ms

MANUEL EVAL (ayrı dosya veya script çıktısı):
- 10 rastgele soruda model cevaplarını kendi değerlendir:
  - 1-5 skala: dilbilgisi, ton, bilgi doğruluğu, format
- Sonuçları ml/README.md'de tablo olarak tut

Yorum (eşik değerler — Türkçe için):
- ROUGE-L < 0.25: adapter öğrenmemiş, lr veya epoch arttır
- ROUGE-L 0.25-0.40: makul, manuel doğrula
- ROUGE-L > 0.40: iyi ama overfitting kontrolü yap (eval-train overlap?)
- BERTScore F1 > 0.75: semantik örtüşme iyi
- BERTScore F1 < 0.60: semantik olarak kötü, base model daha iyi olabilir
```

### Bunu kendin yap:
```bash
python ml/training/evaluate.py
# ROUGE-L ve BERTScore skorları ne?
# 5 örnek yazdı — cevaplar okunabilir mi, Türkçe doğru mu?

# Manuel eval: README.md'ye tablo
# |  # | Soru | Cevap | Dilbilgisi (1-5) | Ton (1-5) | Doğruluk (1-5) |

git add . && git commit -m "eval(p2): ROUGE + BERTScore + manual quality check"
```

---

## Task 5 — Notebook tamamla ⏱ ~45 dk

### Claude Code'a ver:
```
Proje: eduai-platform/ml/

Görev: notebooks/01_data_exploration.ipynb tamamla.

Notebook hücreleri (her biri ayrı):

Hücre 1 (Markdown):
- Başlık: "P2 — Dataset Exploration"
- Amaç: oluşturulan training dataset'ini keşfet ve kalitesini görsel olarak değerlendir

Hücre 2 (Code):
- Veri yükleme (data/processed/train.jsonl)
- İlk 5 örnek göster (pandas DataFrame)

Hücre 3 (Code):
- Temel istatistikler:
  - Toplam örnek sayısı
  - Subject başına sayı
  - Grade başına sayı
  - Ortalama output uzunluğu (karakter + token)

Hücre 4 (Code):
- Bar chart (matplotlib): subject × grade heatmap
- Subject dağılımı dengeli mi görsel olarak

Hücre 5 (Code):
- Histogram (matplotlib): output token uzunlukları
- Outlier var mı (çok kısa/uzun cevaplar)?

Hücre 6 (Code):
- Tokenizer analizi:
  - Tokenizer yükle (AutoTokenizer, config'teki model)
  - En uzun 3 örnek (token sayısına göre) yazdır
  - En kısa 3 örnek yazdır
  - max_seq_length (512) aşan var mı? → dataset clipping gerekli mi

Hücre 7 (Markdown):
- Sonuç ve notlar:
  - Dataset kalitesi gözlemleri
  - Düzeltilmesi gereken noktalar (varsa)
  - Training beklentisi

NOT: nbstripout aktif — commit öncesi output'lar otomatik temizlenir.
Commit sonrası lokalde "Run All" yapıp gözlem için saklayabilirsin.
```

### Bunu kendin yap:
```bash
jupyter notebook ml/notebooks/01_data_exploration.ipynb
# Tüm hücreleri çalıştır, gözlemle

git add . && git commit -m "docs(p2): data exploration notebook"
```

---

## Task 6 — CI pipeline güncelle ⏱ ~30 dk (**yeni — P1'de yoktu**)

**Bu task'ta öğreniyorsun:** Mevcut CI workflow'a job eklemek, ML-specific checks.

### Claude Code'a ver:
```
Proje: eduai-platform/

Görev: .github/workflows/ci.yml dosyasına ml-quality job'ı ekle.

SPEC.md (P2) → "GitHub Actions CI" bölümündeki ml-quality job tanımını uygula.

Yapılacaklar:
1. Mevcut ci.yml'e ml-quality job ekle (lint, test, docker-build'dan sonra):
   - needs: lint (P1 lint ile paralel değil, sonra)
   - Aslında paralel olabilir: lint ve ml-quality bağımsız → iki paralel job,
     sonra test + docker-build bunların sonrasında.
   - Düşün: en mantıklı dependency graph hangisi?

2. ml-quality job içeriği (SPEC'ten):
   - checkout + setup-python 3.11 + cache pip
   - pip install ruff pytest pyyaml jsonschema
   - ruff check ml/ --output-format=github
   - ruff format ml/ --check
   - pytest ml/tests/ -v --tb=short

3. ml/tests/test_data_schema.py yaz (SPEC.md'de tam kod var):
   - JSONL schema validation
   - REQUIRED_KEYS kontrolü
   - VALID_SUBJECTS kontrolü (P1'deki SubjectEnum ile uyumlu!)
   - grade 1-12 kontrolü
   - output min length kontrolü

4. Eğer data/processed/ dosyaları yoksa test'ler pytest.skip() ile geçsin
   (CI'da dataset yok — schema testi sadece varsa çalışsın).
```

### Bunu kendin yap:
```bash
# Lokal test
cd ml && pytest tests/ -v
cd ..
ruff check ml/ && ruff format ml/ --check

git add .github/ ml/tests/ && git commit -m "ci(p2): add ml-quality job with schema validation"
git push
# GitHub Actions: 4 job görmeli (lint + ml-quality + test + docker-build)
```

---

## Task 7 — P3 hand-off dokümanı ⏱ ~30 dk (**yeni — P1'de yoktu**)

**Bu task'ta öğreniyorsun:** Faz geçiş disiplini, artifact dokümantasyonu.

### Claude Code'a ver:
```
Proje: eduai-platform/

Görev: docs/p2/P3_HANDOFF.md yaz — P3 fresh Claude session için kickoff paketi.

İçermesi gerekenler:

1. P2 sonucu — as-of tarih:
   - Hangi base model kullanıldı (final karar)
   - Dataset boyutu (train + eval)
   - En iyi run metric'leri (eval_loss, ROUGE-L, BERTScore)
   - LoRA adapter yolu (ml/models/checkpoints/)
   - MLflow run_id'leri

2. P3 için ne lazım:
   - Adapter nasıl yüklenir (PeftModel.from_pretrained kod örneği)
   - Inference wrapper function imzası (P3 RAG pipeline'da kullanılacak)
   - Base model + tokenizer + adapter tripli
   - Beklenen latency (ms) + VRAM gereksinimi

3. P2'deki bilinçli kabul edilen eksiklikler (varsa):
   - "Product kalitesinde hallucination kontrolü yok → RAG ile çözülecek"
   - "Evaluation ROUGE + BERTScore ile sınırlı → P3'te real user eval"

4. IMPLEMENTATION_NOTES.md referansı (arşiv veya güncel)

5. P3 kickoff prompt taslağı (P1 → P2 handoff'taki gibi kopya-yapıştır metni):
   "Ben Türkçe AI eğitim platformunun P3 (RAG + agent sistemi) fazına
   başlıyoruz. P1 API ve P2 LoRA adapter tamamlandı.
   docs/p2/P3_HANDOFF.md + docs/p3/CONCEPT.md + docs/p3/SPEC.md + docs/p3/TASKS.md okumalısın..."
```

### Bunu kendin yap:
```bash
# docs/p2/P3_HANDOFF.md'yi gözden geçir
# Eksik/belirsiz nokta var mı?

git add docs/p2/ && git commit -m "docs(p2): P3 handoff package"
```

---

## P2 tamamlandı mı? Kontrol listesi

```
[ ] ml/ proje yapısı kuruldu, .gitignore güncel, nbstripout aktif
[ ] data/processed/train.jsonl + eval.jsonl oluşturuldu (min 200 örnek)
[ ] pytest ml/tests/ yeşil (schema validation)
[ ] python training/train.py → adapter models/checkpoints/'e kaydedildi
[ ] mlflow ui → en az 3 deney yan yana karşılaştırılıyor
[ ] ROUGE-L + BERTScore hesaplandı, skorlar kabul edilebilir eşikler içinde
[ ] Manuel eval (10 soru) ml/README.md'de tabloda
[ ] notebook output'lu hali lokalde saklı (commit'e girmemeli — nbstripout)
[ ] GitHub Actions 4 job yeşil (lint + ml-quality + test + docker-build)
[ ] docs/p2/P3_HANDOFF.md yazıldı
[ ] docs/p2/IMPLEMENTATION_NOTES.md bilinçli sapmaları içeriyor
```

---

## P2 → P3 geçişi (P1 pattern'inin tekrarı)

P2 bittiğinde:
1. `docs/p2/SPEC.md`'yi IMPLEMENTATION_NOTES'larla güncelle (inline 📝 callout'lar)
2. `IMPLEMENTATION_NOTES.md` → `IMPLEMENTATION_NOTES_ARCHIVED.md` rename
3. Memory update (reference → archived path)
4. `P3_HANDOFF.md` ile fresh Claude session başlat

---

## Sıkça karşılaşılan sorunlar

**"CUDA out of memory"**
→ `per_device_train_batch_size`'ı 2'ye düşür, `gradient_accumulation_steps`'ı 8'e çıkar (effective batch aynı kalır).

**"TRL AttributeError / unexpected kwarg"**
→ TRL sürüm farkı. `pip show trl` ile kontrol. `SFTTrainer(tokenizer=...)` → `processing_class=...` veya tersi.

**"Trust remote code" interactive prompt**
→ config.yaml'da `trust_remote_code: true` set et veya env `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`.

**"ModuleNotFoundError: bitsandbytes"**
→ Linux/WSL için OK. macOS için bitsandbytes limitli — Colab önerilir.

**MLflow Colab'da silinip duruyor**
→ Tracking URI'yi Google Drive path'ine çek (`/content/drive/MyDrive/mlruns`).

**ROUGE çok düşük ama cevaplar okunabilir**
→ Türkçe morfoloji. BERTScore skoru da düşükse sorun var; BERTScore iyi ama ROUGE düşükse metric artifact'ı, ignore et.

**Eval loss düşüyor ama train loss sabit**
→ Overfit değil ama underfit. Learning rate arttır veya epoch ekle.

**Train loss düşüyor ama eval loss artıyor**
→ Overfitting. Epoch azalt, LoRA dropout artır (0.05 → 0.1), dataset büyüt.
