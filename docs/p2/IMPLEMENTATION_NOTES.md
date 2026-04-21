# P2 — Implementation Notes

> P1 pattern'inin tekrarı: bu dosya P2 boyunca SPEC'ten **bilinçli sapmaları**
> ve nedenlerini kaydeder. Faz sonunda `SPEC.md` inline **📝 Implementation
> note** callout'larıyla güncellenip bu dosya `IMPLEMENTATION_NOTES_ARCHIVED.md`
> olarak rename edilir.

---

## Task 0 — ML proje yapısı (2026-04-20)

### Sapma 1 · Kök `.gitignore` zaten P2-ready
- **SPEC/TASKS beklentisi:** `TASKS.md` Task 0 → kök `.gitignore`'a
  `ml/models/checkpoints/`, `ml/data/raw/`, `ml/mlruns/`, `ml/.env`,
  `notebooks/.ipynb_checkpoints/` eklenecek.
- **Gerçek durum:** Kök `.gitignore` (satır 46-52) P2 hazırlığı sırasında
  önceden bu kuralları içerecek şekilde güncellenmiş. Tüm kurallar mevcut.
- **Aksiyon:** Tekrar eklemedim — duplicate kural yok. `git diff .gitignore`
  boş. Bu aslında "iş zaten yapılmış" durumu, gerçek bir sapma değil ama
  Task 0 checklist'inin boş geçtiğini açıklamak için not düşüldü.

### Sapma 2 · SPEC'teki `ml/data/.gitignore` nested dosyası atlandı
- **SPEC beklentisi (SPEC.md satır 25):** Klasör yapısında
  `ml/data/.gitignore ← raw veriyi ignore et (gizlilik)` var.
- **Neden atlandı:** Kök `.gitignore` zaten `ml/data/raw/` kuralını içeriyor
  (satır 48). Nested `.gitignore` eklemek:
  1. İki farklı yerde aynı kuralı tutmak → ileride tutarsızlık riski
  2. `pre-commit` veya CI kontrollerinde iki farklı `.gitignore` parsing
     maliyeti
- **Aksiyon:** Tek kaynaklı yaklaşım — kök `.gitignore` tek gerçek. Eğer
  ileride `ml/data/raw/` içine bilinçli commit'lenmek istenen bir
  `README.md` / `schema.json` olursa alt `.gitignore` tekrar değerlendirilir.

### Sapma 3 · Boş gitignored klasörler için `.gitkeep` eklenmedi
- **Durum:** `ml/data/raw/` ve `ml/models/checkpoints/` içi gitignored.
  Git boş klasör takip etmediği için clone sonrası bu klasörler yok olacak.
- **Karar:** `.gitkeep` eklemek gitignore istisnası gerektirir
  (`!ml/data/raw/.gitkeep`) — clutter yaratır. Bunun yerine `data_prep.py`
  ve `train.py` içinde `Path(...).mkdir(parents=True, exist_ok=True)`
  pattern'i uygulanacak (Task 1 ve Task 2 scope'u). Kullanıcı klonladığında
  klasörler ilk script çalıştırmada otomatik oluşur.
- **`.gitkeep` eklenen klasörler:** `ml/training/`, `ml/tests/`,
  `ml/data/processed/` — bu üçü Task 1/2/6'da dolacak, yapı git'te görünsün.

### Sapma 4 · requirements.txt birebir SPEC'ten kopyalandı — 2026 sürüm uyarısı
- **Durum:** SPEC'teki pinned sürümler 2024 Q2-Q3 civarı
  (torch 2.3.1, transformers 4.43.3, trl 0.9.6, bitsandbytes 0.43.3).
  Bugün 2026-04-20. PyPI'da hâlâ kurulabilir olmalı ama breaking
  ekosistem değişiklikleri birikmiş olabilir.
- **Karar:** TASKS.md açıkça "SPEC'teki PINNED sürümleri birebir kopyala"
  diyor. Kopyaladım. **Task 2**'de `pip install -r` çalıştırıldığında
  gerçek durum netleşecek.
- **Sonuç:** İlk `pip install` denemesinde torch==2.3.1 bulunamadı → Sapma 5
  ile hemen Task 0 içinde çözüldü (aşağı bak). Bu sapma **Sapma 5 tarafından
  supersede edildi**.

### Sapma 5 · Python 3.13 + modern pin upgrade (2026-04-20)
- **Tetikleyen:** Kullanıcının lokal venv'i Python 3.13. `pip install -r
  ml/requirements.txt` → `torch==2.3.1` için wheel yok hatası:
  ```
  ERROR: Could not find a version that satisfies the requirement torch==2.3.1
  (from versions: 2.6.0, 2.7.0, ..., 2.9.1, 2.10.0, 2.11.0)
  ```
  Torch 2.3.1 (2024-05) Python 3.11 için son yayınlandı; 3.13 için
  wheel bulunmuyor. Bu noktada iki yol vardı:
  1. Python 3.11 venv kurup SPEC pin'lerini korumak
  2. Python 3.13'te kalıp tüm ML stack'i 2026-Q1+ sürümlerine upgrade etmek
- **Seçilen yol:** 2. — **Python 3.13 + modern pin'lere upgrade**.
- **Neden:**
  1. **SPEC bunu öngörmüş:** SPEC.md satır 496-499 *"Upgrade gerekli ise
     sürümü pin'le ve değişiklikleri IMPLEMENTATION_NOTES'a not et"* der.
     Halihazırda blessed yol.
  2. **Ertelenmiş teknik borç:** CONCEPT.md satır 193-212 → Phi-4 Mini /
     Qwen3 7B / LLaMA-3.3 8B gibi 2026 model önerileri yeni
     `transformers` ve `peft` sürümleri gerektirir. 3.11'e dönsek bile
     Task 2 model seçiminde yeniden upgrade baskısı çıkar.
  3. **Öğrenme değeri:** Bu bir öğrenme projesi; 2024 state'ini dondurmak
     yerine 2026 API'larını öğrenmek daha değerli.
- **Yeni strateji — iki dosyalı lockfile pattern'i:**
  - `ml/requirements.txt` → okunabilir floor pin'ler (`>=X,<Y`); yorumlu,
    developer bu dosyayı okuyup anlar
  - `ml/requirements.lock.txt` → `pip freeze` çıktısı; tam pin, CI + eşit
    ortam kurulumu için deterministik
  - Build command (CI + prod): `pip install -r requirements.lock.txt`
  - Development command: `pip install -r requirements.txt` (son stable)
- **Spesifik değişiklikler:**
  | Paket              | SPEC (2024 pin)  | Yeni pin (2026)         | Gerekçe                                   |
  | ------------------ | ---------------- | ----------------------- | ----------------------------------------- |
  | torch              | 2.3.1            | **2.9.1** (exact)       | 3.13 wheel + CUDA uyumluluğu kritik       |
  | transformers       | 4.43.3           | `>=4.46,<5`             | Yeni model arch desteği, generate() API   |
  | datasets           | 2.20.0           | `>=3.0`                 | Streaming iyileştirmeleri                 |
  | peft               | 0.12.0           | `>=0.13`                | LoRA target_modules auto-detect           |
  | trl                | 0.9.6            | `>=0.12`                | SFTTrainer(processing_class=) yeni imza   |
  | accelerate         | 0.33.0           | `>=1.0`                 | 1.x stable API                            |
  | bitsandbytes       | 0.43.3           | `>=0.44 ; !darwin`      | Apple Silicon'da CUDA-only — skip marker  |
  | mlflow             | 2.15.1           | `>=2.17`                | Minor güvenlik/bug-fix                    |
  | diğerleri          | exact            | `>=` floor              | Dev araçları, breaking risk düşük         |
- **`bitsandbytes` macOS durumu:** Apple Silicon'da CUDA olmadığı için
  `bitsandbytes` kurulamaz. `sys_platform != 'darwin'` environment marker
  ile pip bu paketi macOS'te atlar (hata vermez). Training Colab'da
  yapılacağından lokal dev akışını bozmaz. **Colab notebook'a manuel
  `!pip install bitsandbytes` adımı eklenecek** — Task 2 veya Colab
  bölümü güncellemesinde.
- **Task 2'ye downstream etkileri:**
  - `train.py` içinde TRL 0.12+ imzası zorunlu: `SFTTrainer(model=...,
    processing_class=tokenizer, ...)`. Eski `tokenizer=` kullanma.
  - SFTConfig parametrelerinden bazıları taşındı: `max_seq_length`
    artık doğrudan SFTConfig'te değil, `dataset_kwargs` veya `formatting_func`
    aracılığıyla — Task 2 yazarken `help(SFTConfig)` ile doğrula.
  - `BitsAndBytesConfig` parametreleri aynı kaldı (API stable).
- **Reproducibility kaybı kabul edildi:** Floor pin'ler 6-12 ay içinde
  farklı sürümler çekebilir. `requirements.lock.txt` bu riski azaltır
  ama SPEC'teki "tam aynı sürüm garantisi" artık yok. Öğrenme projesi
  için kabul edilebilir bir takas.

---

## Task 1 — Veri seti oluştur (2026-04-20)

### Sapma 6 · Strateji seçimi: Seçenek A + prompt'ta MEB zorunluluğu
- **SPEC/TASKS beklentisi:** Seçenek A/B/C arasından bir strateji seç,
  `ml/README.md`'de dokümante et. Kullanıcı ek olarak
  **"MEB güncel müfredatına uygun olması çok önemli"** şartı koydu.
- **Seçim:** Seçenek A (Sentetik, Claude API).
- **Gerekçe:** `ml/README.md` "Veri stratejisi" bölümünde açık.
- **MEB uyumluluk iki kat enforce edildi** (`data_prep.py:build_prompt`):
  1. System persona: *"Sen Türkiye Milli Eğitim Bakanlığı (MEB) güncel
     öğretim programlarına hakim bir eğitim asistanısın."*
  2. Kurallar #1: *"Sorular MEB güncel müfredatı ile uyumlu, sınıf
     seviyesine uygun olmalı."*
- **Konu seed stratejisi:** SUBJECT_TOPICS dict'i ile her subject × grade
  için 4-5 konu verilip Claude'a expand ettiriliyor. **Spesifik kazanım
  kodları (örn. "9.1.1.1") verilmiyor** çünkü:
  - MEB müfredatı periyodik revize ediliyor; kod formatı dönemsel değişir
  - Claude'un corpus'undaki MEB bilgisi konu-çerçeve seviyesinde güvenilir;
    spesifik kazanım kodları için hallucination riski yüksek

### Sapma 7 · Token sayımı karakter sayımına indirgendi
- **SPEC beklentisi (TASKS Task 1, item 5):**
  *"Ortalama output token uzunluğu (tokenizer'la)"*
- **Gerçek uygulama:** Karakter bazlı ortalama/medyan/min/max
  (`data_prep.py:print_statistics`).
- **Neden:**
  - Phi-3 tokenizer için `transformers` + `trust_remote_code` + model
    download (~500 KB) gerekir
  - `data_prep.py` şu an torch/transformers bağımsız çalışabiliyor
    (sadece `anthropic`, `sklearn`, `dotenv`) — bu hafif kalabalığı
    bozmak istenmedi
  - **Task 5 notebook**'unda (hücre 6) tokenizer analizi zaten var;
    detaylı token dağılımı oraya bırakıldı
- **Etki kabul edilebilir:** Türkçe için Phi karakter/token oranı ~3-4.
  500 karakter ≈ 125-165 token → `max_output_len=1000` karakter filtresi
  zaten 512 token sınırının altında.

### Sapma 8 · Subject × grade kombinasyon kapsama boşlukları
- **SPEC beklentisi:** Subject başına 50 örnek, grade 9-12 dengeli.
- **Gerçek kapsama:** MEB müfredatında **felsefe 9. sınıfta yok** (10-12 var).
  `SUBJECT_TOPICS[subject].get(grade)` None dönerse o (subject, grade) kombinasyonu
  atlanıyor. `data_prep.py:generate_dataset` her subject için `target/len(grades)`
  bölüp min 5 garanti ediyor — dengeli dağılım korunuyor.
- **Beklenen toplam:** 9 subject × 4 grade × ~12 örnek = ~432. Felsefe yoksunluğu
  ve kalite filtresi (~5-10% elenme) ile **~400-420 final kayıt** bekleniyor.

### Sapma 9 · `test_data_schema.py` bu task'ta yazılmadı
- **TASKS Task 1 "Bunu kendin yap":** `cd ml && pytest tests/ -v` istiyor.
- **Sorun:** `test_data_schema.py` SPEC.md'de tam kodu olmasına rağmen
  **TASKS Task 6**'ya yerleştirilmiş. Task 1'de pytest çalıştırılırsa
  "no tests collected" → exit code 5 alır.
- **Karar:** Test dosyasını Task 6'ya bıraktım (TASKS disiplini). Kullanıcıya
  alternatif olarak iki yol sunuldu:
  1. Task 6'ya kadar pytest satırını atla
  2. Test'i şimdi yazmak istersek ml/tests/test_data_schema.py'yi eklemek
     (SPEC'teki kod birebir)
- **Öneri → Karar (2026-04-20):** Test dosyası **Task 1'e çekildi**.
  Kullanıcı veri üretiminden sonra `pytest tests/ -v` çalıştırdığında
  test dosyası olmadığı için exit code 5 aldı. Pragmatik çözüm:
  `ml/tests/test_data_schema.py` Task 1 sonunda yazıldı. Kapsamı:
  - SPEC'teki 4 baseline assertion (required keys, subject enum,
    grade range, min output len)
  - Ek kontrol: **max output len** (1000 char), **boş instruction/output**,
    **input string tipi**
  - `test_no_exact_duplicate_instructions` — dataset içi dup yok
  - `test_train_eval_no_leakage` — split sonrası train/eval ortak
    instruction barındırmamalı (data leakage koruması)
- **Task 6'ya kalan:** Sadece CI wiring (`.github/workflows/ci.yml`'e
  `ml-quality` job eklemek). Test dosyası zaten yerinde.

---

## Task 2 — config.yaml + train.py (2026-04-20)

### Sapma 10 · Chat template hardcoded string yerine `tokenizer.apply_chat_template`
- **SPEC beklentisi (SPEC.md satır 188-193):** `format_prompt` Phi-3 chat
  formatını string olarak hardcoded yazıyor:
  ```python
  f"<|user|>\n{example['instruction']}\n<|end|>\n"
  f"<|assistant|>\n{example['output']}\n<|end|>"
  ```
- **Gerçek uygulama (`train.py:build_formatting_func`):**
  `tokenizer.apply_chat_template(messages, tokenize=False)` kullanıldı.
- **Neden:**
  - Hardcoded template model'e bağımlı. `config.yaml`'da model Phi-4 Mini
    veya Qwen3'e geçilirse string yanlış olur, sessizce bozuk training
    yapar (farklı BOS/EOS → modelin bilmediği pattern).
  - `apply_chat_template` tokenizer'dan model spesifik template okur —
    model-agnostik adaptation.
- **Risk:** Çok eski modellerin tokenizer'ında chat_template yok (Phi-2
  öncesi). Phi-3/4 + Qwen2/3 + LLaMA-3 hepsinde var. Dataset'e etki yok.

### Sapma 11 · SFTConfig `eval_strategy` + `save_strategy` explicit set edildi
- **SPEC beklentisi:** Sadece `eval_steps`, `save_steps` veriyordu.
- **Gerçek uygulama:** `eval_strategy="steps"` + `save_strategy="steps"`
  explicit eklendi.
- **Neden:** Modern `transformers` (4.46+) bazı sürümlerde `eval_steps`
  ayarlı ama `eval_strategy="no"` default ise silently eval atlayabiliyor.
  Explicit strategy → bug-proof.

### Sapma 12 · `load_best_model_at_end` smoke mode'da force-disable
- **SPEC beklentisi:** Config'teki `load_best_model_at_end: true` hep
  geçerli.
- **Gerçek:** `--smoke` modunda `load_best_model_at_end=False`'a zorlanır.
- **Neden:** Smoke 10 step + save_steps=100 → hiç checkpoint oluşmaz;
  `load_best_model_at_end=True` bu durumda trainer.train() exception
  atar. Smoke asıl checkpoint kaydı istemez (pipeline testi).

### Sapma 13 · SIGTERM → KeyboardInterrupt köprüsü
- **Görev beklentisi:** "Keyboard interrupt (Ctrl+C) → adapter save,
  graceful exit"
- **Ek koruma:** Sadece SIGINT değil SIGTERM de yakalanıyor
  (`signal.signal(SIGTERM, ...)` → `KeyboardInterrupt` raise). Çünkü:
  - Colab idle timeout veya OOM killer SIGTERM gönderir
  - Container orchestration (K8s preemption, P4 senaryosu) SIGTERM ile
    öldürür
  - Her iki durumda da 30-60dk'lık training'i kaybetmemek için elde
    olanı diske yazmak kritik.

### Sapma 14 · MLflow `report_to="none"` + manuel callback
- **SPEC beklentisi:** `report_to="none"` + MLflow manuel
- **Uyum:** Aynı; sadece manuel yazımı `StepMetricsCallback` class'ına
  çıkardık (logic merkezi, her step'te sync log).
- **Alternatif düşünülen:** `report_to="mlflow"` TRL integration;
  atıldı çünkü run lifecycle kontrolünü manuel tutmak (`with
  mlflow.start_run()`) artifacts + params + metrics'i tek context'te
  tutuyor. Integration path'inde nested run'lar oluşabilir, debugging
  zor.

### Not · Sürüm kontrolü Mac lokal'de yapılamadı
- **Görev talimatı:** `pip show trl transformers peft` ile sürüm
  doğrulama; API farkı varsa uyarla.
- **Gerçek:** Mac lokal shell'de `trl` modülü yok (`bitsandbytes`
  platform marker ile skip, training Colab'da). Gerçek API doğrulaması
  Colab'da `%pip list | grep -E 'trl|transformers|peft'` ile yapılmalı.
- **Varsayım:** TRL 0.12+ (`processing_class=`), transformers 4.46+,
  peft 0.13+. SPEC + görev talimatındaki tabloya dayalı. Colab'da
  import hatası çıkarsa Task 2 içine ek sapma (15) olarak geri yansıtılır.

### Sapma 15 · TRL 1.0+ API rename: `max_seq_length` → `max_length`
- **Tetikleyen (2026-04-21):** Colab smoke test:
  ```
  TypeError: SFTConfig.__init__() got an unexpected keyword argument 'max_seq_length'
  ```
- **Ortam sürümleri:** trl 1.2.0, transformers 4.57.6, peft 0.18.1
  (bizim varsayımdan — trl 0.12+, tx 4.46+ — oldukça ileride).
- **Doğrulama:**
  ```python
  [p for p in inspect.signature(SFTConfig).parameters if "seq" in p or "length" in p]
  # → ['group_by_length', 'length_column_name', 'max_length']
  ```
  `max_seq_length` parametresi TRL 1.0 breaking release'inde `max_length`
  olarak yeniden adlandırıldı (HuggingFace naming convention uyumu).
- **Fix:** `ml/training/train.py:289` → `max_length=config["model"]["max_length"]`.
  Config tarafındaki key adı (`model.max_length`) zaten `max_length`;
  sadece SFTConfig kwarg adı değişti.
- **Öngörülen sonraki olası breaking (TRL 1.x):** `processing_class=` hâlâ
  destekleniyor ama yumuşak deprecated olabilir; `packing`, `dataset_text_field`
  gibi parametrelerin bazıları 1.x'te `SFTConfig`'e taşınmış olabilir.
  Sıradaki smoke hatası çıkarsa aynı yöntemle:
  `list(inspect.signature(SFTConfig).parameters)` veya
  `list(inspect.signature(SFTTrainer).parameters)` ile kontrol.

### Sapma 16 · T4 BF16 uyumsuzluğu + eksik `prepare_model_for_kbit_training`
- **Tetikleyen (2026-04-21):** Smoke #2:
  ```
  NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda"
  not implemented for 'BFloat16'
  ```
- **Kök sebep:** İki birleşik problem:
  1. **Model dtype:** Phi-3 Mini'nin config'inde `torch_dtype="bfloat16"`.
     `from_pretrained`'e `torch_dtype` override geçirmezsek non-quantized
     katmanlar (embedding, layer_norm, lm_head) bf16'da kalır. T4 (compute
     capability 7.5) BF16 CUDA kernel'lerini destekler **ancak** AMP
     grad scaler'ın `_amp_foreach_non_finite_check_and_unscale_cuda`
     BFloat16 için T4'te implement edilmemiş (kernel sadece FP16 ve
     CC≥8.0 için BF16).
  2. **Eksik QLoRA prep:** SPEC `prepare_model_for_kbit_training` call'unu
     atlamış. Bu standart QLoRA pattern — layer_norm ve output head'i
     fp32'ye promote eder, gradient checkpointing'i aktive eder, input
     embedding'e `require_grad` hook'u ekler. Olmadan 4-bit model üzerinde
     gradient flow bozuk.
- **Fix (`ml/training/train.py:setup_model_and_tokenizer`):**
  ```python
  model = AutoModelForCausalLM.from_pretrained(
      ...,
      torch_dtype=torch.float16,       # T4 uyumluluğu için bf16→fp16 override
  )
  model = prepare_model_for_kbit_training(model)   # QLoRA standart hazırlık
  ```
- **A100/H100 için not:** Bu değişiklik T4-odaklı. A100+ (CC≥8.0) için
  `torch.bfloat16` + config'te `bf16: true` daha iyi numeric stability
  sağlar. Colab Pro A100 kullanılacaksa config.yaml'a bir
  `bf16: true/fp16: false` + train.py'de device capability check
  eklenebilir (Task 3 sweep'ine bırakıldı).
