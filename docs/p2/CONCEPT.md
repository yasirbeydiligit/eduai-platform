# P2 — Model Fabrikası: Konsept Kılavuzu

> P1 bitti, API ayakta. Şimdi o API'ye beyni yapıyoruz.
> Bu dosyayı okumadan SPEC'e geçme.

---

## Büyük resim: ne yapıyoruz?

Hugging Face'den hazır bir model alıyoruz (Phi-3 Mini veya benzeri), bunu Türkçe eğitim verisiyle ince ayar yapıyoruz (fine-tuning), ve MLflow ile her denemeyi kaydediyoruz. Çıktı: Vizyon Koleji'nin ders materyallerine özel cevaplar veren bir model.

---

## Temel kavramlar

### 1. Neden fine-tuning? Base model yetmez mi?

GPT-4 veya Claude gibi modeller genel amaçlı. "Osmanlı'da Tanzimat nedir?" diye sorsan cevap verir ama:
- Senin ders kitabına göre cevap vermez
- Türkçe pedagojik dili kullanmaz
- 9. sınıf seviyesine uygun tonlamayı bilmez
- Vizyon Koleji'nin müfredatını bilmez

Fine-tuning şunu yapar: modeli senin verilerinle "yeniden kalibre" eder. Parametrelerin hepsini değil, küçük bir kısmını.

---

### 2. LoRA nedir? Neden tam fine-tuning yapmıyoruz?

Bir LLM'in milyarlarca parametresi var. Hepsini güncellersen:
- GPU VRAM'i dolup taşar (7B model = ~28GB)
- Eğitim saatler sürer
- "Catastrophic forgetting" riski — model eski bilgisini unutur

**LoRA (Low-Rank Adaptation):** Orijinal ağırlıkları dondurur, yanına çok küçük "adapter" matrisleri ekler. Sadece bu adaptörleri eğitir.

```
Orijinal ağırlık matrisi W (7B parametre) → donduruldu
LoRA adaptörü: A × B (milyonlarca değil, yüz binlerce parametre)
Tahmin sırasında: W + A×B
```

Sonuç: %1-3 kadar parametre eğitiliyor ama performans tam fine-tuning'e yakın.

**QLoRA:** LoRA + quantization. Model 8-bit veya 4-bit'e sıkıştırılır, böylece sıradan GPU'da (16GB VRAM) 7B model bile fine-tune edilebilir.

---

### 3. Training loop nasıl çalışır?

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # 1. Forward pass: model tahmin üretir
        outputs = model(**batch)
        loss = outputs.loss
        
        # 2. Backward pass: hatayı geri yayar
        loss.backward()
        
        # 3. Optimizer adımı: parametreleri günceller
        optimizer.step()
        optimizer.zero_grad()
```

Her iterasyonda:
- Model bir tahmin üretir
- Bu tahmin ile gerçek cevap arasındaki fark (loss) hesaplanır
- Loss, ağırlıklara geri yayılır (backpropagation)
- Ağırlıklar biraz düzeltilir

Bunu binlerce kez tekrarlarsın — model yavaş yavaş "doğru" cevaplar üretmeye başlar.

---

### 4. Dataset formatı: ne tür veri lazım?

Fine-tuning için "instruction tuning" formatı kullanıyoruz:

```json
{
  "instruction": "Tanzimat Fermanı'nın önemini 9. sınıf öğrencisine açıkla.",
  "input": "",
  "output": "Tanzimat Fermanı (1839), Osmanlı'da modern hukukun..."
}
```

Buna **JSONL** formatı denir — her satır bir JSON objesi. Kaç örnek lazım? Fine-tuning için 100-1000 kaliteli örnek yeterli. ChatGPT'yi eğitmek değil, adapte etmek istiyoruz.

---

### 5. MLflow nedir? Neden lazım?

Şu senaryoyu düşün: 3 farklı learning rate denedin, 2 farklı LoRA rank denedin. Hangi kombinasyon en iyi sonucu verdi? Not almadıysan bilmiyorsun.

MLflow şunu yapar:
- Her training run'ı otomatik kaydeder
- Metrikler (loss, accuracy) zaman içinde nasıl değişti → grafik
- Hangi hyperparameter hangi sonucu verdi → tablo
- En iyi modeli "register" et, versiyonla

```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("learning_rate", 2e-4)
    mlflow.log_param("lora_rank", 16)
    
    # training...
    
    mlflow.log_metric("train_loss", loss, step=step)
    mlflow.log_artifact("model_checkpoint/")
```

`mlflow ui` komutuyla tarayıcıda tüm deneyleri görürsün.

---

### 6. Hugging Face ekosistemi

```
transformers  → model yükleme, tokenizer, inference
datasets      → veri yükleme ve işleme
peft          → LoRA/QLoRA adaptörleri
trl           → SFTTrainer (Supervised Fine-Tuning)
accelerate    → multi-GPU desteği
bitsandbytes  → 4-bit quantization (QLoRA için)
```

Bu kütüphaneler birlikte çalışır. Şöyle düşün:
- `datasets` verinizi alır
- `transformers` modeli yükler
- `peft` LoRA adaptörlerini ekler
- `trl.SFTTrainer` hepsini bir araya getirip eğitir

---

### 7. Model seçimi: hangisini kullanacağız?

| Model | Boyut | VRAM | Türkçe | Öneri |
|-------|-------|------|--------|-------|
| Phi-3 Mini | 3.8B | 8GB | Orta | Başlangıç için ideal |
| Mistral 7B | 7B | 16GB | İyi | Google Colab Pro |
| LLaMA-3 8B | 8B | 16GB | İyi | Google Colab Pro |
| Qwen2 7B | 7B | 16GB | Çok iyi | Türkçe için en iyi |

**Öneri:** Phi-3 Mini ile başla (ücretsiz Google Colab'da çalışır), sonra Qwen2 7B'ye geç.

---

### 8. Evaluation: model iyi mi kötü mü?

Training bitti, model nasıl değerlendirirsin?

- **Perplexity:** Modelin ne kadar "şaşırdığını" ölçer. Düşük = iyi.
- **ROUGE skoru:** Üretilen cevap ile referans cevap ne kadar örtüşüyor?
- **Manuel değerlendirme:** En güveniliri. 20-30 soru sor, cevapları kendin değerlendir.

```python
# ROUGE hesaplama
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'])
scores = scorer.score(reference_answer, generated_answer)
```

---

## Başlamadan önce sorular

1. LoRA ile tam fine-tuning arasındaki fark nedir?
2. Neden 4-bit quantization kullanırız?
3. MLflow olmadan iki farklı deneyi nasıl karşılaştırırsın?
4. JSONL formatında bir eğitim örneği yazar mısın?
5. `SFTTrainer` ne işe yarar, neden `Trainer`'dan farklı?
