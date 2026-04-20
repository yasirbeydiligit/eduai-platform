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
