# P4 — Görev Listesi: Content & Tools

> P3 FINALIZED. Fresh session'da P4 başlangıcı için
> [`docs/p3/P4_HANDOFF.md`](../p3/P4_HANDOFF.md) okunmalı.
>
> **Önkoşul:** P3'ün tüm çıktıları çalışıyor (`docker-compose up qdrant -d`,
> `pytest agents/tests/` 22+ PASSED).

---

## Task 0 — Kurulum + ek bağımlılıklar ⏱ ~15 dk

### Claude Code'a ver:
```
Proje: eduai-platform/agents/

Görev: P4 ek dizin ve bağımlılıklar.

1. Yeni klasörler: agents/{content,tools,eval}/ + __init__.py
2. agents/data/{pdfs,osym}/ + .gitignore (PDF'ler repo'ya girmesin)
3. agents/requirements.txt'a ekle:
   sympy>=1.13
   pdfplumber>=0.11
   matplotlib>=3.9
4. pip install -r agents/requirements.txt
5. Smoke: python -c "import sympy, pdfplumber, matplotlib; print('OK')"
```

---

## Task 1 — PDF parser + 5 MEB PDF indeksleme ⏱ ~1.5 saat

**Öğreniyorsun:** PDF parsing kalite gradyanları, structure-aware extraction.

### Claude Code'a ver:
```
Proje: eduai-platform/agents/content/

Görev: pdf_parser.py + 5 örnek MEB PDF indeksle.

1. content/pdf_parser.py: SPEC.md'deki PDFParser.
   - pdfplumber backend
   - detect_quality heuristic: text/page < %60 → "degraded", skip
2. agents/scripts/index_pdfs.py:
   - agents/data/pdfs/*.pdf → DocumentIndexer (eduai_documents)
   - Subject metadata dosya adından (örn. "tarih_9_unite3.pdf" → "tarih")
3. Test data: 5 MEB PDF (tarih, edebiyat, coğrafya, din, felsefe).
   Kullanıcı temin edecek; placeholder OK ilk pas için.
```

### Bunu kendin yap:
```bash
# 5 PDF'i agents/data/pdfs/'a koy
python agents/scripts/index_pdfs.py
git add agents/content agents/scripts/index_pdfs.py
git commit -m "feat(p4): Task 1 — PDF parser + MEB indekleme"
```

---

## Task 2 — ÖSYM loader + 100 örnek ⏱ ~1 saat

### Claude Code'a ver:
```
Proje: eduai-platform/agents/content/

Görev: osym_loader.py + örnek soru bankası.

1. content/osym_loader.py: SPEC.md'deki OSYMLoader.
   - JSON format: {year, exam_type, subject, question, answer_key}
   - Qdrant collection: osym_questions
2. agents/data/osym/yks_2023_sample.json: 10 manuel soru
   (veya kullanıcı resmi ÖSYM JSON parse'lar)
3. Smoke: 100 sample yükle, "x² türevi" sorgusu → en yakın 4 soru
```

---

## Task 3 — Math agent (sympy + LLM + verify) ⏱ ~2 saat

**Öğreniyorsun:** Tool-augmented agents, sympy integration, defensive verification.

### Claude Code'a ver:
```
Proje: eduai-platform/agents/tools/

Görev: math_agent.py + sympy_tool.py + 3 örnek.

1. tools/sympy_tool.py: 3 tool (solve_equation, compute_derivative,
   compute_integral). Her birinin error handling açık.
2. tools/math_agent.py: SPEC.md'deki MathAgent.
   parse → sympy → llm_explain → verify → format
   Disclaimer: "AI çözümü, kontrol et"
   Verification: 2nd LLM (claude-haiku) "Bu çözüm matematik olarak
   doğru mu? Sympy çıktısı [...], LLM açıklaması [...]"
3. tools/router.py: regex classifier
4. agents/scripts/test_math_agent.py: 3 örnek (denklem, türev, integral)
   Beklenen: roots/derivative/integral doğru + LLM açıklama tutarlı
```

---

## Task 4 — Validator + recall@k ⏱ ~1 saat

### Claude Code'a ver:
```
Proje: eduai-platform/agents/

Görev: Sapma 39 follow-up'ları.

1. agents/graph/nodes.py: _WEAK_INDICATORS genişlet (10+ pattern):
   "Maalesef bağlamda... bilgi bulunmamaktadır"
   "kaynaklar yetersiz"
   "veri yok"
   "değerlendirme yapamıyorum"
   (eval_ab_results.csv frequency analysis ile)

2. validator score threshold: top retrieved < 0.7 → konservatif kabul.

3. agents/eval/recall_at_k.py: SPEC.md'deki compute_recall_at_k.
4. agents/data/eval_gold_chunks.csv: 30 sample manuel etiketle
   (eval_idx, gold_source filename).
5. python agents/eval/recall_at_k.py → recall@1/@4/@10
```

### Bunu kendin yap:
- 30 eval Q için doğru chunk dosyası bul (manuel, ~30 dk)
- recall@4 ≥ 0.7 ise retriever sağlam
- recall@4 < 0.5 ise corpus stratejisi yetersiz

---

## Task 5 — API integration + router ⏱ ~45 dk

### Claude Code'a ver:
```
Proje: eduai-platform/services/api/

Görev: /ask/v2'ye math router.

1. routers/questions.py: ask_question_v2'da router.classify():
   - "math" → math_agent.solve()
   - "theory" → mevcut LangGraph
2. dependencies.py: get_math_agent() singleton
3. main.py lifespan: math_agent eager init
4. ENV: MATH_AGENT_ENABLED=true|false
5. Smoke: 2 curl (Türev nedir? + x²-5x+6=0 çöz)
```

---

## Task 6 — Eval full + plots ⏱ ~2 saat

### Claude Code'a ver:
```
Proje: eduai-platform/agents/eval/

Görev: 87 × 4 kondisyon eval.

1. eval/eval_full.py: 87 × 4 (A_baseline, B_rag, C_math, D_hybrid)
2. ROUGE + BERTScore + recall@k (B/D)
3. eval/plots.py: matplotlib bar subject breakdown
4. CSV + PNG: agents/data/eval_full/

Hedef: D_hybrid 80%+ subject'te A_baseline > +0.05 ROUGE-1.
```

### Bunu kendin yap:
```bash
python agents/eval/eval_full.py --n 87
# ~30-45 dk, ~$1-2
```

---

## Task 7 — Testler + finalize ⏱ ~1 saat

```
1. tests/test_pdf_parser.py
2. tests/test_sympy_tool.py
3. tests/test_math_agent.py (mock LLM + sympy)
4. tests/test_router.py
5. tests/test_recall.py

pytest 35+ PASSED beklenir.
```

---

## P4 tamamlandı mı?

```
[ ] 5+ MEB PDF indeksli, retrieve scores > 0.75
[ ] 100+ ÖSYM sorusu osym_questions collection'da
[ ] Math agent 3 örnek doğru + verification log
[ ] Recall@4 ≥ 0.7 (30 etiketli sample)
[ ] D_hybrid eval: A_baseline + 0.05 ROUGE-1
[ ] Validator weak indicator 10+ pattern
[ ] pytest 35+ PASSED
[ ] docs/p4/IMPLEMENTATION_NOTES.md sapmalarla dolu
[ ] docs/p4/SPEC.md inline callout'larla güncel
[ ] docs/p4/P5_HANDOFF.md (Backend Production fresh session için)
```

P4 finalize: NOTES → ARCHIVED rename + memory update + P5'e geç.
