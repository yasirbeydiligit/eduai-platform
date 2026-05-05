# P4 — Content & Tools: Kavramsal Kılavuz

> **As-of:** 2026-05-04 · P3 FINALIZED 2026-05-03 (35+ sapma archived).
> P4 P3 mimarisini gerçek içerikle besler ve matematik için doğru aleti
> (sympy + LLM reasoning) ekler.

---

## P4'ün varlık nedeni — Sapma 39 dersi

P3 sonu 30-sample A/B test ölçtük (RAG-with vs baseline Anthropic):

| Metric | A (baseline) | B (RAG) | Δ |
|---|---:|---:|---:|
| ROUGE-1 F | 0.3753 | 0.3342 | **−0.041** |
| BERTScore F1 | 0.5285 | 0.5178 | **−0.011** |
| B win-rate ROUGE-1 | — | — | **8/30 = %27** |

**Tanı (P3 ARCHIVED Sapma 39):** mimari sağlam, **corpus zayıf**:
1. Train Q&A çiftleri RAG için yanlış format (retriever yanlış chunk getiriyor)
2. Validator weak indicator listesi yetersiz ("Maalesef bağlamda... bilgi yok"
   yakalanmıyor → retry tetiklenmiyor)
3. Modern LLM Türkçe lise sorularında zaten güçlü → RAG noise ekliyor
4. Recall@k metriği hiç ölçülmedi (top-1 score ≠ doğru cevap)

**P4 hedefi:** bu 4 sorunu çöz, eval Δ pozitif yap.

---

## Büyük resim

```
P3 — Mimari (kuruldu)              P4 — Çekirdek değer (yeni)
─────────────────                  ─────────────────────────
LangGraph pipeline                 ✅ MEB konu PDF ingestion (kolay dersler)
RAG retriever                      ✅ ÖSYM soru bankası entegrasyonu
CrewAI multi-agent                 ✅ Math tool agent (sympy + verification)
Anthropic LLM dev backend          ✅ Validator weak indicator genişletme
                                   ✅ Recall@k retrieval metric
                                   ✅ Eval framework genişletme (87×4)
```

P4 sonu: gerçek MEB konu PDF'leri + ÖSYM soruları indeksli, math soruları
sympy ile çözülüyor, eval Δ pozitif.

---

## Temel kavramlar

### 1. İki katmanlı content stratejisi

Tek-corpus (P3) yerine ayrılmış iki katman:

| Katman | Kaynak | Use case | Format |
|---|---|---|---|
| **Theory** | MEB PDF (kolay dersler) | "Türev nedir?", "Tanzimat'ı anlat" | Düz text chunk → Qdrant collection `eduai_documents` |
| **Solved problems** | ÖSYM geçmiş yıl + cevap anahtarı | "Bu soruya benzer örnek" | Soru+anahtar metadata → Qdrant `osym_questions` |

**EBA + Khan eklenmiyor** (Sapma 39 eleştirisi: 3-corpus router karmaşıklığı yarara değmez, YAGNI).

**Math/fizik/kimya işlemli sorular ayrı yol** — RAG değil, **sympy-augmented agent**.

---

### 2. Math Tool Agent — sympy + LLM + verification

**Sebep:** RAG matematik için yanlış araç. "x²+5x+6=0 çöz" sorgusunda
retriever en yakın denklem örneğini getirir → LLM yanlış adım atabilir →
öğrenciye yanlış öğretim → **sorumluluk sorunu**.

**Doğru mimari:**

```
Soru: "x²+5x+6=0 çöz"
   ↓
Math Router (regex: formül/eşitlik/sayı yoğun mu?)
   ↓ math
Math Agent
   ├── parse_question: equation = "x**2+5*x+6=0"
   ├── sympy_tool.solve_equation: roots = [-2, -3]  (deterministik)
   ├── LLM explain (sympy çıktısına ground'lu): adım adım açıklama
   ├── 2nd LLM verify: "bu çözüm matematik olarak doğru mu?"
   └── format: disclaimer "AI çözümü, kontrol et"
```

**Tool stack:** `sympy` (symbolic math, ücretsiz, deterministik) + `numpy`
(sayısal). Wolfram-Alpha **eklenmiyor** (ücretli, MVP için gereksiz).

**Failure mode:** sympy çözemezse (çok karmaşık integral, geometri vs.) →
"Üzgünüm bu soruyu çözemedim, öğretmenle kontrol et" konservatif fallback.

---

### 3. Validator iyileştirme — Sapma 39 follow-up

P3 weak indicator listesi:
```python
_WEAK_INDICATORS = ("bilmiyorum", "yeterli bilgi yok", "emin değil", ...)
```

Eval'da gözlemlenen ek pattern'ler (CSV frequency analysis):
- "Maalesef bağlamda... bilgi bulunmamaktadır"
- "kaynaklar yetersiz"
- "değerlendirme yapamıyorum"
- "veri yok"

**P4'te:**
1. Liste 10+ pattern'e genişlet
2. **Score-based threshold** — top retrieved score < 0.7 → konservatif kabul (retry yok, baseline mantığına dön)
3. (Opsiyonel) NLI-based validator: cevap context'le entail mi? (`dbmdz/bert-base-turkish-cased-snli`)

---

### 4. Recall@k retrieval metric

P3 sadece top-1 score ölçtü. **Yüksek score ≠ doğru cevap** (Sapma 39 kanıtı).
Eksik metrik: **doğru chunk top-k'de var mı?**

**Implementation:**
1. 30 eval Q için **gold chunk** etiketle (manuel veya keyword match)
2. Retriever top-4 sonuçta gold chunk var mı? → Recall@4
3. **Recall yüksek + cevap kötü** → LLM/prompt sorunu
4. **Recall düşük + cevap kötü** → corpus stratejisi yetersiz

Bu metrik Sapma 39'un **kök nedenini** ayırt eder.

---

### 5. Eval framework genişletme

P3 `eval_ab.py`: 30 sample × 2 kondisyon. P4'te:
- 87 örnek tam (eval.jsonl tüm)
- 4 kondisyon: A baseline / B RAG / C math agent / D hybrid (router → A veya C)
- ROUGE + BERTScore + recall@k (B/D)
- Subject breakdown (matematik C kazanır, sözel B kazanır beklenen)

**Hedef:** D_hybrid Δ pozitif **80%+ subject**'te.

---

## P3'ten taşınan dersler

1. **Top-1 score ≠ kalite** — recall@k empirical kanıt için
2. **Doğru aleti seç** — RAG bilgi için, agent+tool reasoning için
3. **Smoke test maliyeti düşük** — her component (PDF parser, sympy) önce smoke
4. **YAGNI** — EBA/Khan eklemeyiz, scope kontrol
5. **Empirical kanıt** — kararlar veri-driven (Sapma 39 modeli)

---

## Telif & yasal

- MEB PDF: deneme/öğrenme aşamasında küçük örnek (~10-20 PDF) OK; **lansman öncesi (P5/P6) hukuki audit zorunlu**
- ÖSYM: kamuya açık resmi arşiv, paylaşım izni var (en güvenli)

---

## Başarı tanımı

P4 başarılı sayılır:
1. Recall@4 ≥ 0.7 (retriever doğru chunk getirebiliyor)
2. D_hybrid ROUGE-1 ≥ A_baseline + 0.05 (router yardım ediyor)
3. Math agent test setinde 80%+ doğru (ÖSYM cevap anahtarı validasyonu)
4. Validator hallucination flag oranı < %5 smoke testte
