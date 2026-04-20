# EduAI — P2: Model Fabrikası

Türkçe lise eğitimi için **QLoRA fine-tuning pipeline'ı**. P1 FastAPI'ye
dokunmaz; bağımsız `ml/` dizininde çalışır. Çıktı: P3 RAG pipeline'ının
arkasına takılacak **LoRA adapter** (~20-100 MB).

> Tam spec: [`docs/p2/SPEC.md`](../docs/p2/SPEC.md)
> Bilinçli sapmalar: [`docs/p2/IMPLEMENTATION_NOTES.md`](../docs/p2/IMPLEMENTATION_NOTES.md)

---

## Veri stratejisi

**Seçim: Seçenek A — Sentetik (Claude API)**

Gerekçe:
- Hızlı iterasyon (~5-10 dk Claude API çağrıları, ~$1-5 maliyet)
- Claude'un kendi training corpus'u Türkiye MEB öğretim programlarını
  geniş çerçevede kapsar
- Açık kaynak TR Q&A dataset'leri (TUQuAD, MKQA-TR) lisans + format
  dönüşüm overhead'i ile gelir
- Hibrit (Seçenek C) kalite artırma isterse sonradan eklenebilir;
  sentetik baseline yeterli başlangıç

**Risk — sentetik veri bias + hallucination taşıyabilir.** `data_prep.py`
bu riski iki katmanda azaltır:

1. **Prompt disiplini** — system persona + explicit rule:
   > "Sen Türkiye MEB güncel öğretim programlarına hakim bir eğitim asistanısın.
   > Kurallar: (1) Sorular MEB güncel müfredatı ile uyumlu olmalı.
   > (4) Emin olmadığın bilgiyi yazma — hallucination yok."

2. **Aşağıdaki manuel QA adımları zorunludur.**

---

## Veri üretimi

```bash
cd ml
cp .env.example .env            # ANTHROPIC_API_KEY'i doldur
python training/data_prep.py    # default: 50 örnek/subject, Haiku model, ~$1
```

**CLI seçenekleri:**

| Flag | Anlam | Örnek |
| ---- | ----- | ----- |
| `--target N` | Subject başına hedef örnek sayısı | `--target 30` |
| `--model {haiku,sonnet,opus}` | Claude model alias'ı | `--model sonnet` |
| `--semantic-dedup` | Ek semantic deduplication (cosine > 0.95) | |
| `--dry-run` | API call yapma; mock dataset ile pipeline testi | |

**Çıktılar:**
- `data/raw/eduai_qa_raw.jsonl` — ham Claude response'ları (gitignored)
- `data/processed/train.jsonl` — eğitim seti (%80)
- `data/processed/eval.jsonl` — değerlendirme seti (%20)

---

## Manuel QA checklist (ZORUNLU — veri üretiminden sonra)

Otomatik filtreler (length, duplicate) kaliteyi garantilemez; insan gözü şart.
`train.jsonl`'den **rastgele 20 örneği** aç ve her biri için kontrol et:

- [ ] **Pedagojik ton:** Cevap öğretici mi, yoksa "AI yanıtı" gibi düz bilgi mi?
- [ ] **Grade uyumluluğu:** {grade}. sınıf seviyesine uygun karmaşıklık?
- [ ] **Türkçe dilbilgisi:** Yazım, imla, cümle yapısı doğru mu?
- [ ] **MEB müfredat uyumu:** Konu {grade}. sınıf {subject} müfredatında mı?
- [ ] **Bilgi doğruluğu:** Bildiğin konularda hatalı iddia var mı?

**Kabul eşiği:** 20 örneğin en az **17'si (%85+)** temiz olmalı. Altında ise
`--model sonnet` veya `--model opus` ile yeniden dene (daha pahalı ama daha
nüanslı); gerekirse prompt'u (`data_prep.py:build_prompt`) revize et.

Gözlemleri bu README'nin altına tablo olarak kaydet (Task 4 manuel eval
tablosuna referans olacak).

---

## Sıkça karşılaşılan sorunlar

| Durum | Çözüm |
| ----- | ----- |
| `ANTHROPIC_API_KEY .env'de bulunamadı` | `cp .env.example .env` → anahtar ekle |
| Parse hatası (tekrar eden) | `--model sonnet` — Haiku bazen markdown fence ekliyor |
| Rate limit | `time.sleep(0.5)` var; yine de hata alırsan API tier'ını kontrol et |
| `bitsandbytes` kurulum hatası (macOS) | Normal — environment marker ile atlanır; training Colab'da |

---

> Training workflow, MLflow kullanımı, Colab adımları ve manuel evaluation
> tablosu **Task 7**'de genişletilecek.
