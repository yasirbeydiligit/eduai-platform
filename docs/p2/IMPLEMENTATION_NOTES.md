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

### Sapma 17 · T4 BF16 hatası tekrarı — canonical QLoRA optimizer şart
- **Tetikleyen (2026-04-21, smoke #3):** Sapma 16'daki fix'ler (`torch_dtype=float16` +
  `prepare_model_for_kbit_training`) **uygulandıktan sonra** aynı hata yine alındı:
  ```
  NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda"
  not implemented for 'BFloat16'
    0% 0/10 [00:10<?, ?it/s]
  ```
- **Kök sebep:** `GradScaler.unscale_()` foreach kernel'i çağırırken grad
  tensor'lerini BFloat16 görüyor. Kaynağı tam belirsiz — muhtemelen HF
  Trainer default `optim="adamw_torch"` + bnb 4-bit dequant pipeline'ının
  etkileşimi; yeni torch 2.10 AMP policy'si de etken olabilir. Kanonik
  çözüm: **optimizer değişikliği**, dtype çözümleri değil.
- **Fix:**
  - `train.py`: SFTConfig'e `optim="paged_adamw_8bit"` eklendi
  - `config.yaml`: `training.optim: "paged_adamw_8bit"` eklendi
  - `train.py`: `model.config.use_cache = False` set edildi (gradient
    checkpointing + cache çatışma warning'i + bazı backward hataları önlenir)
- **Neden `paged_adamw_8bit` çalışıyor:** bitsandbytes optimizer'ı kendi
  grad unscale implementasyonu var, torch AMP foreach kernel'lerini
  kullanmıyor. T4'te BF16 kernel eksikliği bypass oluyor. Ek bonus:
  optimizer state 8-bit → ~%30 VRAM tasarrufu.
- **A100+ için alternatif:** `optim: "paged_adamw_32bit"` — aynı paged
  mekaniği, 32-bit state (daha iyi numeric). T4'te yer almaz ama A100'de
  tercih edilir. Task 3 sweep'inde device-capability-aware seçim düşünülebilir.
- **SPEC'e geri yansıma:** QLoRA setup'ında `optim` açıkça belirtilmeliydi;
  default `adamw_torch` QLoRA için yanlış seçim. SPEC P2 finalize'da
  callout eklenecek.

### Sapma 18 · AMP mixed precision bypass — fp32 training'e düşüldü
- **Tetikleyen (2026-04-21, smoke #4):** Tüm önceki fix'ler
  (`torch_dtype=float16`, `prepare_model_for_kbit_training`,
  `use_cache=False`, `optim=paged_adamw_8bit`) uygulandıktan sonra
  aynı hata persist etti:
  ```
  File ".../torch/amp/grad_scaler.py", line 280, in _unscale_grads_
    torch._amp_foreach_non_finite_check_and_unscale_(...)
  NotImplementedError: ... not implemented for 'BFloat16'
  ```
- **Kök sebep teorisi:** Grad tensor'leri Phi-3 Mini'nin modeling_phi3.py
  içinde bir yerde (muhtemelen attention/RoPE) hard-coded BF16 cast
  görüyor. HuggingFace'in son Phi-3 remote code revision'u
  (`f39ac1d28e925b323eae81227eaba4464caced4e`) bu cast'i içeriyor olmalı.
  `prepare_model_for_kbit_training` sadece top-level parametreleri cast
  eder, ara tensor'leri değil. `paged_adamw_8bit` de torch AMP
  GradScaler'ını bypass etmiyor — bizim önceki hipotezimiz yanlıştı.
- **Fix:** AMP'yi tamamen kapat → saf fp32 training.
  `config.yaml`: `fp16: false, bf16: false`.
- **Maliyet:** ~1.5-2x yavaşlama. Baseline run (66 step, 348 örnek, 3 epoch)
  **14.7 dakika** — T4'te kabul edilebilir. VRAM yaklaşık aynı çünkü
  `prepare_model_for_kbit_training` gradient_checkpointing'i aktive
  etti (aktivasyonlar recompute edilir).
- **Baseline sonucu (MLflow run `1fd87131dcd04114b5068370a51b57d6`):**
  - train_loss: 2.11 → 1.80 (% 15 azalma)
  - mean_token_accuracy: 0.55 → 0.60
  - grad_norm stabil (0.23-0.37)
  - entropy stabil (1.68-1.83)
  - Eval loss **hiç çalışmadı** (eval_steps=100 > max_steps=66).
    Config'te eval_steps=20'ye düşürüldü → Task 3'te 3-4 eval noktası.
- **İleride tekrar fp16 denemek için tetikleyiciler:**
  - Colab Pro A100 — CC 8.0 BF16 hardware desteği
  - TRL veya Phi-3 remote code update'i BF16 cast'i FP16-uyumlu yapması
  - Alternatif model (Qwen3 gibi) BF16 hard-coded değilse
- **Not:** Bu sapma P2 baseline için geçici workaround; P3 inference
  pipeline'ı adapter'ı kendi precision'ıyla yükler, etkilenmez.

### Gözlem · `eval_steps=100` + `max_steps=66` uyumsuzluğu
- Baseline config'teki `eval_steps: 100` Phi-3 + 348 train örnek + 3 epoch
  senaryosunda **hiç eval tetiklemedi** (toplam 66 step). Config
  `eval_steps: 20 + save_steps: 20`'ye güncellendi. Task 3 sweep'inde
  her run'da 3-4 eval noktası olacak; overfitting grafiği (train vs
  eval loss) görülebilecek.

---

## Task 3 — Hyperparameter sweep (2026-04-21 →)

### Deney planı

4 varyant (3 zorunlu + 1 opsiyonel). Her run ~15 dk T4'te, toplam ~60 dk:

| # | Run name | `lora.r` | `lora_alpha` | `lr` | Hipotez |
| - | --- | --- | --- | --- | --- |
| 1 | `lora-r8-lr2e4` | 8 | 16 | 2e-4 | Küçük adapter yeterli mi? (inference hızı + disk avantajı) |
| 2 | `lora-r16-lr2e4` | 16 | 32 | 2e-4 | Yeni config baseline (eval_steps=20 ile) |
| 3 | `lora-r16-lr5e5` | 16 | 32 | 5e-5 | Düşük lr → daha kararlı yakınsama |
| 4 | `lora-r32-lr2e4` | 32 | 64 | 2e-4 | Yüksek kapasite — overfit eder mi? (opsiyonel) |

**Not:** `lora_alpha = 2 * r` otomatik ayarlanır (scale factor `α/r = 2.0`
konvansiyonel sabit). Önceki baseline (`1fd87131…`) `eval_steps=100` ile
koştuğu için eval_loss loglu değil — karşılaştırmada **run #2 yeniden
çalıştırılır** (yeni config'te eval_steps=20).

### CLI override mekaniği
`train.py` üzerine 4 CLI arg eklendi (Sapma 19):
```bash
python training/train.py \
  --lora-r 8 \
  --lr 2e-4 \
  --num-epochs 3 \
  --run-name-suffix "lora-r8-lr2e4"
```
`--lora-r N` verildiğinde `lora_alpha = N * 2` otomatik. Override
verilmeyen değerler config.yaml'dan okunur (in-memory, dosya mutate
etmez — reproducibility korunur).

### Yeni metric'ler (StepMetricsCallback güncellemesi)
- `grad_norm` (step-by-step) — instability/overfitting sinyali
- `mean_token_accuracy` (step-by-step) — pedagojik token tahmin doğruluğu
- `eval_mean_token_accuracy` (eval_steps'te) — generalization

### Sapma 19 · CLI override pattern — 4 ayrı config dosyası yerine
- **Alternatif değerlendirildi:** SPEC/TASKS "4 farklı config dosyası"
  önerdi (`config_r8_lr2e4.yaml`, ...).
- **Seçim:** CLI override. Gerekçe:
  - Tek source of truth (`config.yaml`), varyantlar sadece delta
  - YAML duplication yok — 4 dosya × 45 satır × her commit'te senkron
    tutma zorunluluğu ağır
  - `--run-name-suffix` ile MLflow tarafında yine ayrık run'lar
  - Reproducibility korunur: run parametreleri MLflow'a loglanır
- **Trade-off:** Config dosyaları commit'lenirse fully frozen reproducibility
  sağlar; CLI override'da shell history gerekir. Bu proje bağlamında
  MLflow run params yeterli.

### Sonuç tablosu (2026-04-22, sweep tamamlandı)

| Run | r | α | lr | train_loss (final) | best_eval_loss | duration | MLflow run_id (prefix) |
| --- | - | - | --- | --- | --- | --- | --- |
| lora-r8-lr2e4   | 8  | 16 | 2e-4 | 1.948 | 1.876 | 14.9 dk | `7fc64a25…` |
| lora-r16-lr2e4  | 16 | 32 | 2e-4 | 1.982 | **1.836** | 15.1 dk | `f3dcd9fc…` |
| lora-r16-lr5e5  | 16 | 32 | 5e-5 | 2.065 | 2.019 | 15.1 dk | `ddcfb17a…` |
| lora-r32-lr2e4  | 32 | 64 | 2e-4 | 1.806 | **1.806** | 15.2 dk | `4d862462…` |

### Analiz

**1. Overfitting tespiti: yok, tam tersi hafif underfit.**
- Train-eval gap hepsinde küçük (~0.03-0.04) ve **aynı yönde** (ikisi de
  düşüyor).
- r=32: train (1.806) ≈ eval (1.806) → mükemmel generalization, gap ≈ 0.
- r=16: train (1.982) / eval (1.836) → eval train'den **daha düşük**;
  dropout (0.05) + gradient checkpointing kaynaklı regularization etkisi,
  normal.
- Tüm curve'ler step 60'ta hâlâ düşüş trendinde — plateau yok → **daha
  fazla epoch (5-6) fayda sağlayabilir** (Task 4 sonrası karar noktası).

**2. lr=5e-5 başarısızlığı:**
- Learning rate 4x küçük, 3 epoch'ta parametre hareketi yetersiz.
- Eval loss 2.019 — diğer run'lardan 0.18+ kötü; underfit klasiği.
- 5+ epoch'la yakalayabilirdi ama 3-epoch sweep kriterinde eleniyor.
- **Öğrenme:** Phi-3 Mini QLoRA için lr tabanı 2e-4 civarı; 5e-5 agresif
  düşük. Gelecek varyant denemeleri 1e-4 ile 3e-4 aralığında kalmalı.

**3. r=32 vs r=16 — cost/quality trade-off:**

| Kriter | r=32 | r=16 | Seçilen |
| --- | --- | --- | --- |
| best_eval_loss | 1.806 | 1.836 | r=32 (Δ 0.03) |
| Trainable params | ~56M | ~28M | r=16 (2x küçük) |
| Adapter disk | ~112 MB | ~56 MB | r=16 |
| Inference throughput | baseline | +~1% | r=16 |
| Overfit riski (future) | ↑ | ↓ | r=16 |

**Karar: r=16 lr=2e-4 pareto-optimal.** 0.03 eval_loss delta ROUGE/
BERTScore'da pratik kalite farkı üretmeyecek (<%1 beklenti). r=16 adapter
2x küçük, P3 RAG inference için daha iyi. r=32 ancak Task 4 evaluation
sonrası r=16 yetersiz çıkarsa tercih edilir.

### 🏆 Task 4 için seçilen adapter

- **Run:** `lora-r16-lr2e4`
- **MLflow run_id prefix:** `f3dcd9fc…`
- **Adapter yolu:** `/content/drive/MyDrive/eduai_checkpoints` (en son
  `load_best_model_at_end=True` ile yüklenmiş checkpoint)
- **Beklenen baseline:** eval_loss 1.836 → ROUGE-L ~0.30-0.40 (Türkçe
  morfoloji sınırı), BERTScore F1 ~0.70-0.80

### Sapma 20 · SPEC'teki "opsiyonel r=32 sweep" anlamlı katkı vermedi
- **SPEC/TASKS beklentisi:** r=32 "yüksek kapasite" olarak sunuldu,
  opsiyonel bırakıldı.
- **Sweep sonucu:** r=32'nin katkısı Δ=0.03 eval_loss — pratik anlamda
  ihmal edilebilir. Ama **overfitting riski olmadığını** doğruladı
  (train ≈ eval), bu da dataset'in kapasite darboğazı yaratmadığı
  anlamına gelir (dataset daha fazla veri ile iyileşebilir → Task 4
  sonrası augmentation kararı).
- **Gelecek deneyler için:** r=32 + 5 epoch + lr=1e-4 kombinasyonu
  "production quality" adayı olabilir. Task 4 sonrası fine-tune sweep
  kararına bırakıldı.

---

## Task 4 — Evaluation (2026-04-22 →)

### Hedef
Seçilen adapter'ın (r=16 lr=2e-4) Türkçe pedagojik cevap üretme
kalitesini ROUGE + BERTScore + manuel QA ile ölç; P3 RAG'a
entegrasyon öncesi "yeterince iyi mi?" kararını ver.

### evaluate.py tasarımı
- Base Phi-3 Mini + LoRA adapter yükler (4-bit quantize + PeftModel)
- `eval.jsonl`'den **ilk 20 örnek** (deterministic — seed 42 ile sıralı)
- Her instruction için `do_sample=False` + `max_new_tokens=256` → tekrarlanabilir
- Metric seti:
  - **ROUGE-1 / ROUGE-L** (rouge-score, use_stemmer=False) — Türkçe
    morfolojide zayıf, yine de karşılaştırma noktası
  - **BERTScore F1** (lang="tr" → xlm-roberta-large) — semantik örtüşme,
    Türkçe için güvenilir sinyal
  - **Avg inference latency** (ms) — P3 RAG pipeline'ında throughput
    tahmini için
- 5 örnek yan yana yazdırır (Soru/Referans/Üretilen/ROUGE-L) — manuel
  denetim için
- Sonuçlar MLflow'a **ayrı evaluation run**'ı olarak: `evaluation_rouge1`,
  `evaluation_rougeL`, `evaluation_bertscore_f1`,
  `evaluation_avg_inference_ms`. Training run'larını kirletmez,
  `tags={"type": "evaluation"}` ile filtrelenebilir.

### Sapma 21 · SPEC'teki `format_prompt` raw template yerine
- **SPEC (evaluate.py satır 376):** Hardcoded Phi-3 chat template
  ```python
  prompt = f"<|user|>\n{instruction}\n<|end|>\n<|assistant|>\n"
  ```
- **Uygulama:** `tokenizer.apply_chat_template(messages,
  add_generation_prompt=True)`. Train.py'deki Sapma 10 ile tutarlı —
  model değişirse format otomatik adapte olur.

### Sapma 22 · Tokenizer önce adapter dizininden yüklenir
- **SPEC:** Tokenizer config.yaml'daki base model adından yükleniyor.
- **Değişiklik:** Önce `adapter_dir/tokenizer_config.json` kontrol edilir;
  varsa oradan. Nedenleri:
  - Training sırasında `tokenizer.save_pretrained(output_dir)` ile
    adapter yanına kaydedildi (pad_token eşitlemesi dahil)
  - Adapter portable — base model HF'den yeniden çekilse bile tokenizer
    aynı state'te kalır
  - Fallback: adapter'da yoksa base model adından

### Sapma 23 · BERTScore model indirme uyarısı (operasyonel)
- **Ekleme:** `compute_metrics` içinde ilk BERTScore çağrısından önce
  konsola "model indirilir" bilgi notu basılıyor.
- **Neden:** İlk eval Colab session'da ~1.5GB xlm-roberta-large
  indirme → kullanıcı "takıldı mı?" diye kuşkulanabilir. Operasyonel UX.

### Sapma 24 · Inference'ta `trust_remote_code=False` + eager attention
- **Tetikleyen (2026-04-22):** İlk eval çalıştırması:
  ```
  AttributeError: 'DynamicCache' object has no attribute 'seen_tokens'
  File ".../modeling_phi3.py", line 1291, in prepare_inputs_for_generation
    past_length = past_key_values.seen_tokens
  ```
- **Kök sebep:** Microsoft'un Phi-3 HF Hub deposundaki custom
  `modeling_phi3.py` (commit `f39ac1d2…`) eski transformers API
  kullanıyor — `DynamicCache.seen_tokens` attribute'u transformers 4.50+
  sürümde kaldırıldı (yeni API: `DynamicCache.get_seq_length()`).
  Remote Phi-3 code uzun süredir güncellenmemiş.
- **Fix (`evaluate.py:load_adapter`):**
  ```python
  trust_remote_code=False      # Native transformers Phi-3 code path
  attn_implementation="eager"  # SDPA/flash cache uyumsuzluğu önleme
  ```
  transformers 4.40+ Phi-3 için native implementasyon içeriyor; model
  weights aynı, sadece Python code path değişiyor.
- **Neden training'de hata yok?** Training forward-only (generation yok).
  Inference'ta `model.generate()` `prepare_inputs_for_generation`
  call eder → broken custom code path tetiklenir.
- **Training tarafı ne olacak?** Training'de `trust_remote_code=True`
  kaldı — fine-tuning çalışıyor, degişiklik gereksiz risk. Sadece
  inference (evaluate.py + P3 RAG) için native path.
- **P3'e etki:** P3 inference wrapper'ı da `trust_remote_code=False`
  + `attn_implementation="eager"` kullanmalı. P3_HANDOFF.md'ye not düşülecek.

### Sapma 25 · `torch_dtype` → `dtype` rename (transformers 4.57)
- **Tetikleyen:** Eval başlangıcında warning:
  `torch_dtype is deprecated! Use dtype instead!`
- **Fix:** Her iki yerde (`train.py:setup_model_and_tokenizer`,
  `evaluate.py:load_adapter`) `torch_dtype=` → `dtype=`.
- **Etki:** Yalnızca uyarı susturmak; davranış aynı. `torch_dtype`
  hâlâ çalışıyor ama deprecated; temizlik için güncelleme.

### Sapma 26 · Colab'da torch'u **reinstall etmeyin** — torchvision mismatch
- **Tetikleyen (2026-04-22):** Evaluation import'unda:
  ```
  RuntimeError: operator torchvision::nms does not exist
  ModuleNotFoundError: Could not import module 'PreTrainedModel'
  ```
  Import zinciri: `peft` → `transformers` → `transformers.image_utils`
  → `torchvision` → `torch.library.register_fake("torchvision::nms")`
  **FAIL** (torch ↔ torchvision operator mismatch).
- **Kök sebep:** Colab fresh session `torch 2.10.0+cu128` + `torchvision
  0.25.0+cu128` uyumlu ikili ile gelir (CUDA 12.8 build tag'li). Bizim
  erken oturumda `pip install -q torch==2.10.0 --upgrade` komutu
  **PyPI vanilla 2.10.0** (farklı CUDA build) ile native'i değiştirdi.
  Operator ABI uyumsuzluğu import'ta patladı.
- **Neden training'de görülmedi?** Training import chain'i
  `peft → transformers.modeling_utils` (text modeling) ile sınırlıydı;
  `transformers.image_utils` tetiklenmedi → torchvision import
  edilmedi → mismatch görünmedi. Evaluation'da `peft` init chain
  tam load → fail.
- **Fix (Colab tek-hücre template'te):** requirements.txt'ten **torch
  satırını filter out et**, Colab native torch+torchvision+torchaudio
  ikilisini bozma:
  ```bash
  grep -v '^torch==' ml/requirements.txt > /tmp/colab_req.txt
  pip install -q -r /tmp/colab_req.txt
  pip install -q bitsandbytes
  ```
- **Durumu kurtarma:** Mevcut session bozulduysa `Runtime > Disconnect
  and delete runtime` → fresh runtime. Yeniden install "torch upgrade"
  içermediği için tutarlı kalır.
- **Kalıcı çözüm için ileri dönüş:** requirements.txt'te torch
  pin'ini `sys_platform == 'linux'` marker'ıyla Colab için skip
  edilir hale getirilebilir (macOS dev için hâlâ pinli). Task 6
  CI wiring aşamasında değerlendirilecek.

### Sonuç tablosu (eval çalıştıktan sonra doldurulacak)

| Metric | Değer | Eşik | Durum |
| --- | --- | --- | --- |
| ROUGE-1 (avg) | _TBD_ | — | — |
| ROUGE-L (avg) | _TBD_ | 0.25-0.40 makul, >0.40 iyi | _TBD_ |
| BERTScore F1 | _TBD_ | 0.60-0.75 orta, >0.75 iyi | _TBD_ |
| Avg inference (ms) | _TBD_ | < 3000 ms makul (T4) | _TBD_ |
| Manuel QA (avg /20) | _TBD_ | ≥ 15 (75%) geçer | _TBD_ |

### Karar ağacı (eval sonrası)

```
ROUGE-L ≥ 0.35 + BERTScore ≥ 0.75 + Manuel QA ≥ 15/20
  → Taşı P3'e (Task 5-7 tamamla, P2 FINALIZE)

ROUGE-L 0.25-0.35 + Manuel QA 12-15/20
  → Task 3.5: 5 epoch sweep OR dataset 435 → 800 büyütme

ROUGE-L < 0.25 OR Manuel QA < 12/20
  → Dataset kalite/prompt revize (Task 1'e kısmi geri dönüş)

Hallucination var ama otomatik metrikler iyi
  → P3 RAG context-grounded cevapla düzeltir; P2 yeterli
```
