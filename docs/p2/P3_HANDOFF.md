# P2 → P3 Hand-off

> Fresh Claude session P3'e başlarken bu dosya + `docs/p3/CONCEPT.md` +
> `docs/p3/SPEC.md` + `docs/p3/TASKS.md` okunmalı. P2 boyunca biriken
> teknik kararlar ve karakter/disiplin transferi burada.

**As-of:** 2026-04-28
**P2 Status:** Implementation tamam; **finalize öncesi** (SPEC inline callout
güncellemesi + IMPLEMENTATION_NOTES archive rename Task 7 commit'i sonrasında
yapılır — adımlar bu dosyanın sonunda).

---

## 1. P2 Durumu — Özet

### Tamamlandı (Task 0 → Task 6)

| # | Task | Çıktı |
| - | ---- | ----- |
| 0 | ML proje yapısı | `ml/` ağacı, requirements.txt (relaxed pinler), .env.example |
| 1 | Dataset | 435 örnek Türkçe Q&A (348 train + 87 eval), MEB müfredat odaklı |
| 2 | Training pipeline | `train.py` (TRL 1.2 + paged_adamw_8bit + apply_chat_template) |
| 3 | Hyperparameter sweep | 4 run Phi-3 üzerinde — r=16/lr=2e-4 pareto-optimal |
| 4 | Evaluation | `evaluate.py` (ROUGE + BERTScore + 5 sample print) |
| 5 | Notebook | `01_data_exploration.ipynb` 7 hücre EDA |
| 6 | CI | 4-job pipeline: `lint`, `ml-quality`, `test`, `docker-build` |

### Final Konfigürasyon

| Parametre | Değer | Karar Kaynağı |
| --- | --- | --- |
| **Base model** | `Qwen/Qwen3-4B-Instruct-2507` | Sapma 27→28 (Phi-3 Türkçe yetersiz) |
| **LoRA r / α** | 16 / 32 | Task 3 sweep — pareto cost/quality |
| **Target modules** | `q_proj, k_proj, v_proj, o_proj` | attention-only baseline |
| **Quantization** | 4-bit nf4 + double quant | T4 (16GB) sığması için |
| **Optimizer** | `paged_adamw_8bit` | Sapma 17 (T4 BF16 grad scaler bug bypass) |
| **Precision** | fp32 (`fp16: false, bf16: false`) | Sapma 18 (T4 + Qwen3 AMP sorunu) |
| **Epochs / LR** | 3 / 2e-4 | Cosine schedule, warmup 0.03 |
| **Effective batch** | 16 (4 × 4 grad accum) | T4 VRAM marjı |
| **max_seq_length** | 512 | tokenizer analizi truncation yok dedi |

### Final Metrikler

**Training (MLflow run `c7878a569c85493dbd45cc5a4a0efbc6`):**
- train_loss: 1.96 → **1.45**
- eval_loss: 1.43 → **1.347** (best, plateau yok — daha fazla epoch'la inebilir)
- mean_token_accuracy: 0.62 → 0.69
- grad_norm stabil (0.35-0.48)
- Süre: 14 dk T4

**Evaluation (MLflow run `01822d480ec1471b90357c1643dd6aba`):**
- ROUGE-1: 0.399
- ROUGE-L: 0.249 (eşik dibinde — Türkçe morfolojisi yapısal düşüş)
- BERTScore F1 (tr): 0.640 (orta)
- Avg inference latency: 22978 ms (T4 + 4-bit + use_cache default)

### Phi-3 Baseline (referans — "ne yapmamalı")

P2 başlangıçta **Phi-3 Mini** ile denendi, 4 hyperparameter sweep + 1 evaluation
çalıştı. Sonuç bozuk Türkçe (kelime salatası, döngüsel tekrar). Detaylar:
IMPLEMENTATION_NOTES Sapma 27.

| Phi-3 sonucu | Değer | Karşılaştırma |
| --- | --- | --- |
| ROUGE-L | 0.183 | Qwen3'te +%36 |
| BERTScore F1 | 0.530 | Qwen3'te +%21 |
| Üretim kalitesi | "yapılan birçok süreç..." döngü | — |

MLflow Phi-3 run'ları silinmedi; **gelecekte aynı hatayı yapmamak için
referans değerli.**

---

## 2. P2 Çıktıları — Inventory (P3'ün tüketeceği)

| Artifact | Konum | Tipik kullanım |
| --- | --- | --- |
| **LoRA adapter** | `/content/drive/MyDrive/eduai_qwen3-4b-instruct-2507_ckpt/` (Colab Drive) | P3 inference wrapper'da yüklenir |
| Adapter dosyaları | `adapter_model.safetensors`, `adapter_config.json`, `tokenizer*` | PeftModel.from_pretrained input |
| Training config | `ml/training/config.yaml` | Reference |
| Eval set | `ml/data/processed/eval.jsonl` (87 örnek) | P3 RAG-with vs RAG-without A/B testi |
| Train set | `ml/data/processed/train.jsonl` (348 örnek) | P3'te hot-set olarak Qdrant'a indekslenebilir |
| MLflow runs | `/content/drive/MyDrive/eduai_mlruns/` | Geçmiş deney karşılaştırma |

**Adapter taşıma seçenekleri** (P3 lokal vs Colab dev):
1. **Colab'da kal:** P3 development de Colab'da → adapter Drive yolundan direkt yüklenir (en kolay)
2. **HF Hub'a push:** `huggingface-cli upload <kullanıcı>/eduai-tr-adapter ml/models/checkpoints/`
   → P3'ten `PeftModel.from_pretrained("kullanıcı/eduai-tr-adapter")`
3. **Lokal'e indir:** Drive Web UI ZIP indir → `~/Desktop/eduai_platfrom/ml/models/checkpoints/`
   → P3 lokal embedding + Colab inference endpoint kombinasyonu

**P3 başlangıcında karar verilecek.** Default tavsiye: **2 (HF Hub)** — versioned,
shareable, P3 RAG service public/private erişebilir.

---

## 3. P3 Inference Wrapper — Adapter Yükleme

### Tam yükleme kod örneği

```python
"""
P3 inference wrapper — base model + LoRA adapter + tokenizer üçlüsü.
P2 evaluate.py'deki load_adapter fonksiyonu temel alınır; aynı setup
P3 RAG pipeline'ında reuse edilir.
"""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Sabitler — P3 config'inden okunabilir
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_PATH = "/content/drive/MyDrive/eduai_qwen3-4b-instruct-2507_ckpt"
# Veya HF Hub yolu: "username/eduai-tr-adapter"

# 4-bit quantization — T4 uyumlu, A100'de fp16 native daha hızlı
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Base model — trust_remote_code=False (Qwen native, custom code gerekmez)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16,         # transformers 4.57: torch_dtype → dtype
    trust_remote_code=False,
    # attn_implementation default (SDPA) — Qwen için optimal
)

# LoRA adapter bağla; merge etmez, forward'da paralel çalışır
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# Tokenizer — adapter dizininden tercih (training'de save edildi)
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

### Önerilen P3 inference wrapper imzası

P3 RAG pipeline'ı bu fonksiyonu çağıracak — context-grounded cevap üretimi:

```python
def generate_answer(
    instruction: str,
    context: str | None = None,        # RAG retrieved doküman içeriği
    max_new_tokens: int = 256,
    temperature: float = 0.0,           # 0 = deterministic (eval'da), 0.3-0.7 production
    use_cache: bool = True,             # fast path; T4'te DynamicCache uyarı verirse False
) -> tuple[str, dict]:
    """
    Args:
        instruction: Kullanıcı sorusu (Türkçe)
        context: RAG'dan gelen doküman snippet'leri; None ise saf model cevabı
        max_new_tokens: Üretim limiti
        temperature: 0 = greedy, >0 = sampling
        use_cache: KV cache (default True; bazı model+transformers kombinasyonlarında False gerek)

    Returns:
        (text, metadata):
            text: model cevabı (special token'lar atılmış)
            metadata: {
                "latency_ms": float,
                "input_tokens": int,
                "output_tokens": int,
                "context_used": bool,
            }
    """
    # Context varsa system prompt'a ek
    if context:
        system_msg = (
            "Aşağıdaki bağlamdan yararlanarak Türkçe pedagojik bir cevap ver. "
            "Bağlamda olmayan bilgi uydurma.\n\n"
            f"BAĞLAM:\n{context}"
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": instruction},
        ]
    else:
        messages = [{"role": "user", "content": instruction}]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    # ... generation loop
```

**Tam implementation P3 Task 1'de yazılır;** P2 `evaluate.py:generate_answer`
fonksiyonu temel alınabilir (orada context yok, P3'te eklenir).

---

## 4. Beklenen Performans — VRAM ve Latency

### Hardware Profili

| Setup | VRAM (load) | VRAM (peak inference) | Latency (256 tok) |
| --- | --- | --- | --- |
| T4 (Colab free) | ~3.5 GB | ~6-8 GB | **~22-25 sn** ← P2 ölçümü |
| A100 40GB (Colab Pro) | ~3.5 GB | ~6-8 GB | **~3-5 sn** (fp16 native) |
| L4 (Colab Pro) | ~3.5 GB | ~6-8 GB | ~8-12 sn |
| Lokal CPU | ~5 GB RAM | ~6 GB | **300+ sn** (production değil) |

### P3 RAG Latency Etkisi

RAG context input token sayısını **3-5x artırır** (örn. 50 → 200 token user
question + 1500 token retrieved context). T4'te:
- Saf P2 inference: ~22 sn
- RAG context'li (~1500 token in): **+%30-50 → ~30-35 sn**

Bu Production için kabul edilemez. P3'te optimization yolları:
1. **vLLM serving** (~10x hızlandırma — page attention + batching)
2. **Speculative decoding** (~%30 hızlandırma)
3. **Adapter merge** (`PeftModel.merge_and_unload()` — ~%10 hızlandırma + simplification)
4. **Quantization upgrade** (4-bit → 8-bit veya AWQ — quality vs speed trade-off)
5. **Streaming response** (kullanıcı algılaması — ilk token 1-2 sn'de)

P3 Task'larında bu seçenekler değerlendirilmeli.

---

## 5. Bilinçli Kabul Edilen Eksiklikler (P3 için context)

### Kalite Kapsamı

1. **Bilgi doğruluğu hataları** ("Türkiye petrol zengini" gibi hallucinations)
   → **Beklenen P3 çözümü:** RAG retrieved döküman context'i → model alıntılayarak
   cevap verir, ezberden değil. CONCEPT.md § 1 bu mimariyi açıklıyor.

2. **Manuel QA atlandı (Yol B kararı, 2026-04-28)**
   → P3 sonrası **A/B karşılaştırması** (RAG-with vs RAG-without): 10 rastgele
   soruda subjective 1-5 skala. Bu gerçek kullanım senaryosunu yansıtır;
   saf P2 manuel QA değildi.

3. **Evaluation 20 örnekle sınırlı**
   → İstatistiksel anlamlılık zayıf. P3'te full eval set (87 örnek) + RAG
   variant'ları ile genişletilecek.

### Teknik Borçlar

4. **Inference latency T4'te 22 sn**
   → P3'te vLLM veya A100 üzerinde production serving denenmeli.

5. **fp32 training (T4 BF16 grad scaler bug bypass)**
   → A100'de fp16/bf16 retest mümkün (~30% hızlandırma + biraz iyileşmiş kalite).
   Sapma 18 detay.

6. **Phi-3 macerası MLflow'da kayıtlı ama silinmedi**
   → P3 baseline'ında **Qwen3 run'ı kullanılır**, Phi-3 run'ları "ne yapmamalı"
   referansı. MLflow tag `mlflow_phase=p3` ile P3 deneyleri ayrılır.

7. **P2 `data_prep.py` Claude API'si sentetik veri üretti**
   → P3 RAG için **gerçek ders kitabı** veya **Türkçe MMLU/açık dataset** gerekecek.
   Sentetik verinin "AI gibi yazıyor" tonu zaten P2'de gözlendi.

8. **`requirements.txt` floor pin (`>=`) kullanıyor; lock dosyası yok**
   → P3'te eğer reproducibility kritik olursa `requirements.lock.txt` (pip freeze)
   eklenebilir. P2'de Sapma 5 maliyet/değer dengesi sebebiyle atlandı.

---

## 6. P2 Boyunca Doğrulanan Pattern'ler (P3'e taşınmalı)

### Kod Disiplini

| Pattern | Yanlış | Doğru | Kaynak |
| --- | --- | --- | --- |
| transformers dtype | `torch_dtype=torch.float16` | `dtype=torch.float16` | Sapma 25 (transformers 4.57 rename) |
| Phi-3 + 4-bit | greedy decode | repetition_penalty + sampling | Phi-3 macerası (terkedildi) |
| Qwen + native | `trust_remote_code=True` | `trust_remote_code=False` | Sapma 24 (custom code DynamicCache bug) |
| QLoRA training | `optim="adamw_torch"` (default) | `optim="paged_adamw_8bit"` | Sapma 17 (T4 BF16 scaler) |
| QLoRA prep | `from_pretrained` direkt LoRA'ya | `prepare_model_for_kbit_training` ekle | Sapma 16 (gradient flow) |
| Chat formatlama | hardcoded `<\|user\|>...` | `tokenizer.apply_chat_template` | Sapma 10 (model-agnostik) |
| Pydantic mutable default | `Field(default=[])` | `Field(default_factory=list)` | P1 carry-over |

### Süreç Disiplini (sample inspection > metrics-only)

1. **Loss düşüşü ≠ kalite iyileşmesi.** Phi-3 baseline loss 2.11→1.80 "iyi"
   görünüyordu, üretim çöp. **5-sample print eval döngüsünün parçası** olmalı.

2. **Differential diagnosis.** Bug çıktığında:
   - Suspect izole et (fine-tuning mi base mi mantığı)
   - Hipotez test et (config toggle)
   - Tek değişiklik → tek test
   - Bulgular değiştiğinde sıradaki hipoteze geç

3. **Smoke test maliyeti düşük, değeri yüksek.** Base model 3-5 dk smoke
   test'i 25 dk başarısız training'i önler. Model değişimi öncesi şart.

4. **Spec'teki uyarıları küçümseme.** CONCEPT.md "Phi-3 Türkçe: Orta"
   yazıyordu — biz "yeterli" varsaydık, 2-3 iterasyon kaybettik. **İyi
   belgelenmiş projelerde kararlar erkenden değerlendirilir.**

5. **Pareto disiplini.** İlk %80 kalite %20 efor ile gelir. Kalan %20
   marjinal kazanç için %80 efor + risk var. **"Yeterince iyi"yi
   tanı, doğru aleti seç** (style → fine-tuning, doğruluk → RAG).

### Iş Akışı (P1 + P2'de doğrulandı, P3'te tekrarlanır)

```
Fresh session → CONCEPT.md → SPEC.md → TASKS.md sırayla görevler
Her task → çalışma sırasında sapmaları IMPLEMENTATION_NOTES.md'ye yaz
Faz sonu → SPEC.md'yi callout'larla güncelle → NOTES → ARCHIVED rename
        → PN_HANDOFF.md kickoff paketi yaz
```

---

## 7. User Tercihleri (P1 + P2'de doğrulandı)

Bu tercihler memory'de (`feedback_code_style.md`, `user_role.md`) zaten kayıtlı
ama P3 fresh session'da explicit:

- **Türkçe iletişim** + **Türkçe açıklayıcı kod yorumları**.
- **Spec'i körü körüne uygulama** — mantıksız nokta görüldüğünde gerekçeyle düzelt,
  sapmayı IMPLEMENTATION_NOTES'e yaz. P2'de **31 bilinçli sapma** kayıtlı.
- **Kod düzenli + okunaklı** — docstring, mantıksal gruplamaya ayırıcı yorum.
- **Yorumlar WHY anlatır, WHAT değil** (well-named identifier zaten WHAT'ı veriyor).
- **CI/test/lint disiplini commit öncesi şart**: `ruff check + ruff format --check
  + pytest`. Hata pre-commit hook olmadığında manuel doğrulama.
- **Risk-yönetimli aksiyonlar:** smoke test → full training, fallback path
  daima açık (model değişimi durumlarında özellikle).

---

## 8. P3 Mimarisi — P2'nin Yeri

```
┌─────────────────────────────────────────────────────────┐
│  P3 RAG + Agent Pipeline                                │
│                                                         │
│   User → LangGraph routing                             │
│             ↓                                           │
│         Qdrant retriever (Türkçe embedding)            │
│             ↓                                           │
│         Retrieved context + user question              │
│             ↓                                           │
│   ┌─────────────────────────────────────┐              │
│   │  P2 Adapter (style/format provider) │ ← BİZİM YER │
│   │  Qwen3-4B + LoRA r=16               │              │
│   └─────────────────────────────────────┘              │
│             ↓                                           │
│         Validator (CrewAI agent)                       │
│             ↓                                           │
│   P1 API (FastAPI) → User                              │
└─────────────────────────────────────────────────────────┘
```

**P2'nin P3'teki rolü:** **form/style/ton sağlayıcı.** Bilgi doğruluğu
retriever'dan gelir; biz pedagojik tonu, formatı, akıcılığı sağlıyoruz.

---

## 9. Referanslar

| Doküman | İçerik |
| --- | --- |
| `docs/p2/SPEC.md` | P2 spesifikasyonu (Task 7 sonrası inline callout'larla güncel) |
| `docs/p2/IMPLEMENTATION_NOTES.md` | Tüm sapmalar (31 madde), Task 0 → 6 |
| `docs/p2/CONCEPT.md` | Kavramsal arka plan (LoRA, QLoRA, MLflow, eval metrikleri) |
| `docs/p2/TASKS.md` | Görev listesi (P2'nin başında) |
| `ml/README.md` | ML pipeline kullanım klavuzu |
| `ml/training/config.yaml` | Reference config (Qwen3 final) |
| `ml/training/evaluate.py` | Inference + metric pattern (P3 reuse) |

---

## 10. P3 Kickoff Prompt Taslağı

Aşağıdaki metni **fresh Claude Code session'a ilk mesaj olarak** ver. P1 → P2
handoff'taki pattern'i takip ediyor + P2 boyunca öğrenilenler eklendi.

```markdown
> Ben senior seviyeye doğru ilerleyen bir Python backend geliştiricisiyim.
> Türkçe konuşuyorum, kod yorumlarını Türkçe ve açıklayıcı bekliyorum.
>
> **EduAI Platform** adlı 4 fazlı AI eğitim platformu öğrenme projemin
> **P3 (RAG + agent sistemi)** fazına başlıyoruz. P1 FastAPI backend
> (2026-04-18) ve P2 LoRA adapter (Qwen3-4B-Instruct-2507, 2026-04-28)
> tamamlandı. Kod GitHub'da: `yasirbeydiligit/eduai-platform`.
>
> Lütfen şu sırayla oku:
> 1. `docs/p2/P3_HANDOFF.md` — P2'den P3'e geçiş bağlamı (bu dosya)
> 2. `docs/p3/CONCEPT.md` — P3'ün kavramsal arka planı
> 3. `docs/p3/SPEC.md` — P3'ün implementation spec'i
> 4. `docs/p3/TASKS.md` — adım adım görev listesi
>
> İsterseniz bağlam için `docs/p4/GUIDE.md`'a da göz atabilirsiniz.
>
> **Karakter notları (P1 + P2'de doğrulandı):**
> - Spec'i körü körüne uygulama — mantıksız/eskimiş noktaları gerekçesiyle
>   düzelt, sessizce yapma. P2'de 31 bilinçli sapma kayıtlı; bu pattern devam.
> - Her sapmayı `docs/p3/IMPLEMENTATION_NOTES.md`'ye yaz.
> - Loss düşüşü ≠ kalite iyileşmesi: sample inspection eval döngüsünün
>   parçası (P2'de Phi-3 macerası bunu öğretti).
> - Smoke test maliyeti düşük, değeri yüksek (3-5 dk yatırım).
> - Doğru aleti seç: style/format → fine-tuning, doğruluk → RAG (P2 dersi).
> - CI gate disiplini: `ruff check + ruff format --check + pytest`
>   commit öncesi şart.
>
> **P2 çıktıları (kullanılacak):**
> - LoRA adapter: `Drive/eduai_qwen3-4b-instruct-2507_ckpt` veya HF Hub'a push
> - Base model: `Qwen/Qwen3-4B-Instruct-2507`
> - Eval set: `ml/data/processed/eval.jsonl` (87 örnek, A/B test için)
>
> **Task 0'dan başla, spec'i bana ver, sonra uygulayalım.**
>
> Memory'deki dosyalar zaten yüklü olacak: `project_eduai.md`, `user_role.md`,
> `feedback_code_style.md`, `reference_p1_implementation_notes.md`,
> `reference_p2_implementation_notes.md` (P2 finalize sonrası eklenecek).
```

---

## 11. P2 Finalize Adımları (Task 7 commit sonrası, P3'e geçmeden önce)

P1 pattern'inin tekrarı — bu adımlar **kullanıcı tarafından** yapılacak:

### A. SPEC inline callout güncellemesi
`docs/p2/SPEC.md` dosyasını IMPLEMENTATION_NOTES'taki sapmalarla güncelle.
P1'deki gibi, her sapma SPEC'in ilgili bölümünde **📝 Implementation note**
callout'u olarak görünmeli. Örnek:

```markdown
## Training pipeline (config.yaml)

> **📝 Implementation note (Sapma 18):** SPEC'te `fp16: true` yazıyor ama
> T4 + Qwen3 kombinasyonunda BF16 grad scaler bug'ı sebebiyle fp32'ye
> düşüldü. Detay: IMPLEMENTATION_NOTES_ARCHIVED.md.
```

Otomatik bir script yok — manuel review + edit. Yaklaşık 1-2 saat efor.

### B. NOTES → ARCHIVED rename
```bash
git mv docs/p2/IMPLEMENTATION_NOTES.md docs/p2/IMPLEMENTATION_NOTES_ARCHIVED.md
```
Bu dosya **dondurulmuş tarihçe** olur; P3'te eklenecek yeni sapmalar
`docs/p3/IMPLEMENTATION_NOTES.md` (taze dosya) içine yazılır.

### C. Memory update
`memory/project_eduai.md`'yi P2 FINALIZE notuyla güncelle. Yeni reference
memory: `memory/reference_p2_implementation_notes.md` (P1 örneği gibi —
ARCHIVED yoluna pointer + P2 öğrenilenler özeti).

### D. P2 Finalize commit
```bash
git add docs/p2/ memory/
git commit -m "chore(p2): finalize P2 phase — SPEC callouts, NOTES archive, memory update"
git push
```

### E. P3 hazır
Yeni Claude Code session aç, yukarıdaki prompt taslağını yapıştır, P3 başlasın.

---

**Hand-off tamamlandı.** Bu dokümanın değeri sadece teknik bilgi değil; P2'de
biriken disiplin (sapma yazımı, smoke test, model seçim eleştirisi) P3'te de
sürmeli. Her şey gibi süreç de iteratiftir.
