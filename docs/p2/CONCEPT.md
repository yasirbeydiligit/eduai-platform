# P2 — Model Fabrikası: Konsept Kılavuzu

> P1 bitti, API ayakta. Şimdi o API'ye beyni yapıyoruz.
> Bu dosyayı okumadan SPEC'e geçme.

---

## Büyük resim: ne yapıyoruz? Ne yapmıyoruz?

**Yapıyoruz:** Açık kaynak küçük-orta boy bir dil modelini (Phi-3/4, Qwen2.5/3, LLaMA-3.2/3.3 ailesinden biri) Türkçe eğitim örnekleriyle **QLoRA ile fine-tune** ediyoruz. MLflow ile her deneyi kayıtlı tutuyoruz. Çıktı: P3'te RAG + agent sisteminin arkasına takılacak bir **LoRA adapter**.

**Yapmıyoruz (bilinçli olarak):**
- "Tam fine-tuning" (tüm parametreleri güncellemek) — gerek yok, maliyetli, katastrofik unutma riski
- Sıfırdan model eğitmek (bu akademik bir yıl + milyon dolar)
- P2 tek başına "ürün kalitesinde bir asistan" üretmek — bu P1+P2+**P3** birlikte oluşur
- Gerçek öğrenci verisi kullanmak — gizlilik riski, sentetik/açık kaynak veri kullanıyoruz

**Gerçekçi başarı tanımı:**
> P2 sonunda elimde, P3 RAG pipeline'ının arkasına taktığımda Türkçe eğitim sorusuna **pedagojik tonla** cevap veren, instruction-following yeteneği güçlenmiş bir **LoRA adapter** var.

Bu, "model her şeyi doğru bilsin" değil, "tonu ve formatı bize uygun olsun" hedefi. **Bilgi doğruluğu** için asıl mekanizma P3'teki RAG (retrieval) — fine-tuning değil.

---

## Temel kavramlar

### 1. Neden fine-tuning? Base model yetmez mi?

GPT-4 / Claude gibi kapalı modeller bu proje için uygun değil: maliyet, gizlilik, özelleştirme esnekliği yok. Açık kaynak base model'ler (Phi-3/Qwen2) şunları **yapar**:
- Genel Türkçe gramer/semantik (özellikle Qwen)
- Basit instruction-following

Ama şunları **yapmaz**:
- Senin ders kitabına göre cevap vermez (context yok)
- Türkçe pedagojik dili kullanmaz (formatı bilmiyor)
- Belirli sınıf seviyesine uygun tonlama yapmaz
- "Adım adım çözüm" gibi domain-specific yapıyı izlemez

**Fine-tuning'in iki farklı amacı olabilir:**
1. **Knowledge injection** — modele yeni *gerçekler* öğretmek. Zor, çok veri ister (5000+ örnek). P2'nin hedefi değil.
2. **Style/format alignment** — modelin *nasıl* yanıt verdiğini değiştirmek. 200-500 örnekle mümkün. **P2'nin hedefi.**

Bilgi doğruluğu için RAG (P3) kullanıyoruz — model ders kitabından alıntıyla cevap verir, ezberden değil.

---

### 2. LoRA nedir? Neden tam fine-tuning yapmıyoruz?

Bir LLM'in milyarlarca parametresi var. Hepsini güncellersen:
- GPU VRAM'i dolup taşar (7B model = ~28GB fp16, ~56GB fp32)
- Eğitim saatler/günler sürer
- **Catastrophic forgetting** — model genel yeteneklerini unutur

**LoRA (Low-Rank Adaptation):** Orijinal ağırlıkları dondurur, yanına çok küçük "adapter" matrisleri ekler. Sadece bu adaptörleri eğitir.

```
Orijinal ağırlık matrisi W (7B parametre)     → donduruldu
LoRA adaptörü: ΔW ≈ A × B                      → eğitilecek (~0.1-1% parametre)
Tahmin sırasında: h = Wx + (A × B)x = (W + AB)x
```

**Matematiksel sezgi:** Büyük matrisi W (örn. 4096×4096 = 16M parametre) yerine iki küçük matris A (4096×r) ve B (r×4096) ile yaklaşık ifade ediyoruz. `r` (rank) küçükse (8, 16, 32), toplam parametre dramatik düşer. Denemeler gösterdi ki düşük-rank update'ler çoğu adapted task için yeterli.

**QLoRA:** LoRA + quantization. Model 8-bit veya 4-bit'e sıkıştırılır, böylece sıradan GPU'da (16GB VRAM) 7B model bile fine-tune edilebilir. Quantization sadece forward pass'te; LoRA adapter full precision kalır.

**`r` (rank) seçimi:**
- 8 → minimalist, hızlı, düşük kapasite
- 16 → dengeli (P2 için iyi başlangıç)
- 32-64 → yüksek kapasite, yavaş, overfitting riski (küçük dataset'te)

**`lora_alpha`:** adapter'ın etkisini ölçekler, genelde `r*2`. Scale factor `alpha/r`.

**Target modules:** hangi attention projeksiyonlarına LoRA uygulanacak. `q_proj, v_proj` minimum; tam set `q_proj, k_proj, v_proj, o_proj + gate_proj, up_proj, down_proj` (MLP dahil). Daha fazla target = daha kapsamlı adaptation.

---

### 3. Training dynamics — parametre güncelleme döngüsü

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        outputs = model(**batch)                    # forward
        loss = outputs.loss                         # cross-entropy
        loss.backward()                             # gradient hesabı
        optimizer.step()                            # parametre update
        optimizer.zero_grad()                       # gradient sıfırla
        scheduler.step()                            # learning rate update
```

**Anlaman gereken alt-kavramlar:**

- **Learning rate (`lr`):** her adımda parametreleri ne kadar hareket ettireceğiz. Fine-tuning için genelde `1e-4` ile `5e-4` arası. Çok yüksek → instability, çok düşük → öğrenmiyor.

- **Warmup:** ilk birkaç step'te lr'yi 0'dan hedef değere yumuşakça çıkarmak. Birdenbire yüksek lr ile başlamak model'i bozabilir. `warmup_ratio: 0.03` = toplam adımların %3'ü warmup.

- **LR scheduler (cosine):** training boyunca lr'yi cosine eğrisiyle düşür. Son epoch'ta lr ≈ 0 olur → fine convergence. Alternatifler: linear, constant.

- **Gradient accumulation:** mini-batch küçükse (VRAM sınırı), N adımda bir optimizer.step() yap. "Effective batch size" = `per_device_batch × grad_accum_steps`. Örn. `batch=4, grad_accum=4` → effective 16.

- **Epochs:** tüm dataset üzerinden kaç geçiş. Fine-tuning için 2-5 arası yaygın. Çok fazla → overfit; çok az → underfit.

- **Checkpoint stratejisi:** her N step'te disk'e adapter kaydet. "Best model" için eval loss'a göre en iyisini tut. Disk doldurmamak için `save_total_limit: 2` gibi.

- **Early stopping (opsiyonel):** eval loss K epoch boyunca düşmezse durdur. Aşırı training'i engeller.

---

### 4. Dataset formatı: ne tür veri lazım?

Fine-tuning için **instruction tuning** formatı:

```json
{
  "instruction": "Tanzimat Fermanı'nın önemini 9. sınıf öğrencisine açıkla.",
  "input": "",
  "output": "Tanzimat Fermanı (1839), Osmanlı'da modern hukukun temellerini attı...",
  "subject": "tarih",
  "grade": 9
}
```

Buna **JSONL** formatı denir — her satır bir JSON objesi.

**Data quality > data quantity.**
- 500 kaliteli örnek > 5000 gürültülü örnek
- "Kaliteli" ne demek: doğru bilgi, pedagojik ton, tutarlı format, uygun uzunluk (output genelde 2-5 cümle)

**Üretim stratejileri (P2 için karar vermemiz gereken):**
1. **Sentetik (Claude/GPT-4 ile):** Hızlı, ölçeklenir ama "AI-ürettiği gibi" tona eğilim var. Kalite QA manuel gerekli.
2. **Açık kaynak TR dataset'ler:** TUQuAD, MKQA-TR, Turkish NLI — direkt eğitime uygun format'ta değil, dönüştürme gerekir.
3. **Hibrit:** Sentetik seed + manuel edit pass.

P2 SPEC sentetik öneriyor; alternatifleri açık tutuyoruz.

**Train/eval split:**
- %80/%20 yaygın. Düşük dataset'te (200-500) overlap riski için:
  - **Stratified split:** her `subject` ve `grade` kombinasyonu hem train'de hem eval'da temsil edilsin
  - **Deduplication:** semantic benzerlik ile duplicate kontrolü (sentence-transformers ile cosine sim)

---

### 5. MLflow — neden lazım?

3 farklı learning rate × 2 farklı `lora_r` = 6 deney. Not almadan hangisi en iyi bilinmez.

MLflow:
- Her **run** otomatik kayıt (parametreler, metrikler, artifact'ler)
- Metric grafikleri zaman içinde (loss curve)
- Hyperparameter karşılaştırma tablosu
- **Model registry**: en iyi modeli "register" et, versiyonla, stage'le (Staging/Production)

```python
import mlflow

with mlflow.start_run(run_name="phi3-lora-r16-lr2e4"):
    mlflow.log_params({"lr": 2e-4, "lora_r": 16})
    # training...
    mlflow.log_metric("train_loss", loss, step=step)
    mlflow.log_artifact("model_checkpoint/")
```

`mlflow ui --port 5000` → tarayıcıda tüm deneyleri görüntüle + karşılaştır.

**Colab not:** MLflow Colab'da lokal (`mlruns/` geçici), notebook yeniden başlatılınca silinir. Çözüm: Google Drive'a mount + `tracking_uri=/content/drive/MyDrive/mlruns`.

---

### 6. Hugging Face ekosistemi

```
transformers      → model yükleme, tokenizer, inference
datasets          → veri yükleme ve işleme
peft              → LoRA/QLoRA adaptörleri
trl               → SFTTrainer (Supervised Fine-Tuning wrapper)
accelerate        → multi-GPU + mixed precision
bitsandbytes      → 4-bit/8-bit quantization (QLoRA için)
evaluate          → metric hesaplama (ROUGE, BERTScore vs.)
```

Birbirine bağlı çalışır:
- `datasets` verinizi alır ve iterate edilir formata sokar
- `transformers` modeli + tokenizer yükler
- `bitsandbytes` modeli 4-bit'e quantize eder
- `peft` donmuş modele LoRA adapter ekler
- `trl.SFTTrainer` tüm loop'u çalıştırır, `accelerate` altyapısıyla

**⚠️ Sürüm hassasiyeti:** Bu kütüphaneler **çok hızlı gelişiyor**. `trl`'deki `SFTConfig`/`SFTTrainer` API'si neredeyse her minor release'te değişiyor. Kodunu yazarken `pip show trl transformers peft` ile sürümü kontrol et, breaking change varsa SPEC kodunu adapte et (P1'deki "eleştirel düşün" prensibi).

---

### 7. Model seçimi: 2026'da neler var?

| Model | Boyut | VRAM (QLoRA) | Türkçe | Lisans | Öneri |
|-------|-------|--------------|--------|--------|-------|
| **Phi-3 Mini (4k)** | 3.8B | 6-8 GB | Orta | MIT | ✅ Başlangıç, Colab T4 |
| **Phi-4 Mini** | 3.8B | 6-8 GB | Orta-iyi | MIT | ⭐ 2026 alternatif |
| Qwen2.5 7B | 7B | 10-12 GB | İyi | Tongyi | Colab Pro |
| **Qwen3 7B/14B** | 7-14B | 12-20 GB | Çok iyi | Apache 2.0 | ⭐ Türkçe için 2026 en iyi |
| Mistral 7B v0.3 | 7B | 12 GB | Orta | Apache 2.0 | Alternatif |
| LLaMA-3.3 8B | 8B | 12 GB | İyi | LLaMA License | Alternatif |

**2026 reality check:**
- Phi-3 Mini hâlâ çalışır ama **Phi-4 Mini** (aynı boyut, daha iyi) veya **Qwen3 7B** (Türkçe'de belirgin üstün) tercih edilmeli.
- SPEC bir başlangıç önerir; kullanıcı HuggingFace'te güncel durumu kontrol edip en uygun modeli seçebilir.
- Lisans kontrol et (özellikle LLaMA ve Tongyi ticari kısıtları var).

**Öneri sıralaması (Colab T4 için):**
1. Phi-4 Mini (eğer çıktıysa) — modern + küçük
2. Qwen3 7B — Türkçe için
3. Phi-3 Mini — fallback, kesin çalışır

---

### 8. Evaluation: model iyi mi kötü mü?

Training bitti, nasıl ölçeriz?

**Otomatik metrikler:**
- **Perplexity:** modelin ne kadar "şaşırdığı". Düşük = iyi. Ama fine-tuning'de ezberlemeye eğilimli → yanıltıcı olabilir.
- **ROUGE (1/L):** üretilen cevap ile referans arasında n-gram/LCS overlap. Hızlı. **Sınırı:** Türkçe morfolojisinde zayıf — "gidiyordum" ve "gitmekteydim" düşük ROUGE verir, aynı anlamdadır.
- **BERTScore:** referans + üretilen cevabı embedding'e çevirir, cosine similarity hesaplar. Türkçe için **çok daha iyi** (BERTurk veya xlm-roberta-large base). Yavaş.
- **LLM-as-judge (opsiyonel):** Claude/GPT-4'e "bu iki cevabı kıyasla, hangisi daha iyi?" diye sormak. Pahalı ama insan'a yakın.

**Manuel değerlendirme (zorunlu):** otomatik metrikler eksiktir. 20-30 çeşitli soru sor, cevapları kendin oku:
- Türkçe dilbilgisi doğru mu?
- Pedagojik ton uygun mu?
- Bilgi doğruluğu (bildiğin konularda)
- Format tutarlı mı (örn. "adım adım çözüm" isterken öyle mi dönüyor)

```python
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'])
rouge = scorer.score(reference, generated)

P, R, F1 = bert_score_fn([generated], [reference], lang="tr", verbose=False)
bert_f1 = F1.mean().item()
```

---

### 9. Reproducibility — seed ve sürüm

Training stochastic: random init, batch shuffling, dropout. Aynı kodu iki kez çalıştırsan farklı sonuç alırsın.

Reproducibility için:
```python
from transformers import set_seed
set_seed(42)  # torch + numpy + python random hepsini set eder
```

Ayrıca `config.yaml`'de sabit seed değerini sakla. Bu MLflow comparison gürültüsünü azaltır.

Sürüm pinleme: `requirements.txt`'te `>=` yerine `==` tercih edilir — 6 ay sonra aynı kodu çalıştırdığında aynı davranışı almak için. trl/transformers gibi hızlı değişen paketlerde kritik.

---

### 10. Safety, bias, lisans

Fine-tuning önemsiz bir süreç değil:

- **Bias amplification:** training datan'daki tonlama/ideoloji modelde güçlenir. Sentetik veride dikkat et.
- **Safety regressions:** base model'in refusal/safety davranışları fine-tuning ile zayıflayabilir. Küçük instruction dataset'te özellikle.
- **Hallucination:** fine-tuning bilgi doğruluğunu garanti etmez. "Güvenli görünen yanlış cevap" riski yüksek. RAG (P3) bunu hafifletir.
- **Lisans uyumluluğu:** base model lisansı çıktıyı da bağlar. Qwen2/3 Apache 2.0 → serbest. LLaMA → Meta lisansı. Phi-3/4 → MIT.

**Production'a yaklaşıyorsan:** evaluation suite'ine safety prompts ekle ("bombay nasıl yapılır", "özel kişi hakkında ifşa" gibi) ve model'in mantıklı refuse ettiğini doğrula.

---

## Başlamadan önce sorular (self-check)

1. LoRA ile tam fine-tuning arasındaki fark nedir? Ne zaman hangisi?
2. QLoRA'da neden 4-bit? Doğruluğu etkiler mi?
3. Warmup + cosine scheduler ne işe yarar, neden constant lr değil?
4. Gradient accumulation ne zaman gerekir?
5. MLflow olmadan iki farklı deneyi nasıl karşılaştırırsın?
6. Ezberleyen bir model (overfitting) nasıl tespit edilir? Train loss vs eval loss grafiği ne söyler?
7. JSONL formatında pedagojik ton taşıyan bir örnek yaz.
8. `SFTTrainer` ne işe yarar, `Trainer`'dan farkı nedir?
9. ROUGE Türkçe için neden zayıf, BERTScore neden daha iyi?
10. "Modelim iyi eğitildi" dediğinde hangi 3 sinyale bakıyorsun?
