# P3 → P4 Hand-off

> Fresh Claude session P4'e başlarken bu dosya + `docs/p4/CONCEPT.md` +
> `docs/p4/SPEC.md` + `docs/p4/TASKS.md` okunmalı. P3 boyunca biriken
> teknik kararlar, eval bulgusu, karakter/disiplin transferi burada.
>
> **As-of:** 2026-05-04 · P3 FINALIZED 2026-05-03 (35+ sapma archived)

---

## 1. P3 Durumu — Özet

### Tamamlandı (Task 0 → 6 + Task 4 perf + Task 5 eval)

| # | Task | Çıktı |
|---|---|---|
| 0 | Yapı + Qdrant smoke | `agents/` ağaç + Qdrant container + smoke test |
| 1 | RAG embeddings + indexer | `intfloat/multilingual-e5-large` (F-1 ÇÖZÜLDÜ Sapma 7) + dedup + zengin metadata |
| 2 | RAG retriever | LangChain Document + score metadata + subject filter |
| 3 | LangGraph pipeline | retrieve → generate → validate → retry/format → END + Anthropic dev backend |
| 4 | CrewAI multi-agent | Researcher + Writer (1.14.3, F-2 ÇÖZÜLDÜ Sapma 22) + post-validator |
| 5 | P1 API entegrasyonu | `/v1/documents/upload` + `/v1/questions/ask/v2` + lifespan eager init |
| 6 | Testler | 22 PASSED in-memory Qdrant + FakeEmbedder + MockLLM |
| **+** | **Task 4 perf** | LRU embed cache + Qdrant compat fix + post-validator + SSE streaming endpoint |
| **+** | **Task 5 eval A/B** | 30 sample → **Δ negatif** (RAG bu setup'ta zarar veriyor) — Sapma 39 kritik bulgu |

### Final Stack

| Bileşen | Değer |
|---|---|
| **Embedding** | `intfloat/multilingual-e5-large` 1024-dim (e5 prefix auto) |
| **Vector DB** | Qdrant `v1.12.4` Docker (Sapma 1, sabit pin) |
| **Chunking** | `RecursiveCharacterTextSplitter` 500 char + Türkçe separators |
| **LLM (dev)** | Anthropic `claude-haiku-4-5` |
| **LLM (prod path)** | Qwen3-4B + LoRA via vLLM (P5'te etkin — stub şu an) |
| **Orchestration** | LangGraph compiled StateGraph |
| **Multi-agent** | CrewAI 1.14.3 (Researcher + Writer, sequential) |
| **API** | FastAPI `/ask/v2` + `/ask/v2/stream` SSE |
| **Test** | in-memory Qdrant + FakeEmbedder + MockLLM, 22 PASSED |

### Kritik Empirical Bulgu — Sapma 39 (P4'ün varlık nedeni)

**30 sample A/B (claude-haiku baseline vs LangGraph RAG):**

| Metric | A baseline | B RAG | Δ |
|---|---:|---:|---:|
| ROUGE-1 F | 0.3753 | 0.3342 | **−0.041** |
| BERTScore F1 | 0.5285 | 0.5178 | **−0.011** |
| B win-rate ROUGE-1 | — | — | **8/30 = %27** |

**Tanı (4 hipotez):**
1. Train Q&A çiftleri RAG için yanlış format — retriever yakın ama yanlış chunk getiriyor
2. Validator weak indicator listesi yetersiz ("Maalesef bağlamda... bilgi yok" yakalanmıyor)
3. Modern LLM Türkçe lise sorularında zaten güçlü → RAG noise ekliyor
4. Recall@k metriği eksik — top-1 score ≠ doğru cevap

**P4 hedefi:** 4 sorunu çöz, eval Δ pozitif yap.

---

## 2. P3 Çıktıları — P4'ün tüketeceği

| Artifact | Konum | P4 kullanımı |
|---|---|---|
| **Qdrant collection** `eduai_documents` | localhost:6333 | P4'te yeni MEB PDF'leri eklenecek |
| **TurkishEmbedder** | `agents/rag/embeddings.py` | + LRU cache (Sapma 36) — değiştirme |
| **DocumentIndexer** | `agents/rag/indexer.py` | source_name parametresi (Sapma 32) — kullan |
| **EduRetriever** | `agents/rag/retriever.py` | client DI (Sapma 33) — test'lerde önemli |
| **LangGraph pipeline** | `agents/graph/pipeline.py` | router pre-step ile genişlet |
| **CrewAI tools/validators** | `agents/crew/` | Solved problems agent için ek role olabilir |
| **Eval CSV** | `agents/data/eval_ab_results.csv` | Weak indicator pattern frequency analysis kaynak |
| **Test fixtures** | `agents/tests/conftest.py` | FakeEmbedder/MockLLM pattern devam |

---

## 3. P4'ün Yapacağı İş — Özetle

### A. Content pipeline (Task 1-2)
- **MEB PDF ingestion**: 5+ konu PDF (kolay dersler — tarih, edebiyat, coğrafya, din, felsefe, biyoloji)
- **ÖSYM loader**: 100+ geçmiş soru + cevap anahtarı → `osym_questions` collection
- **Skip:** matematik/fizik/kimya PDF (image-heavy, math agent yolu)

### B. Math tool agent (Task 3)
- **sympy** + **LLM reasoning** + **2nd LLM verification** + **disclaimer**
- 3 tool: solve_equation, compute_derivative, compute_integral
- LangGraph subgraph veya CrewAI Crew (kendi seç implementation'da)
- Router: regex-based (formül/sayı yoğun → math, aksi → theory/RAG)

### C. Validator iyileştirme (Task 4)
- Weak indicator listesi 10+ pattern (eval CSV'den çıkar)
- Score-based threshold: top retrieved < 0.7 → konservatif kabul
- (Opsiyonel) NLI validator: `dbmdz/bert-base-turkish-cased-snli`

### D. Recall@k metric (Task 4)
- 30 eval Q için gold chunk manuel etiketle
- Recall@1, @4, @10 ölç
- recall@4 ≥ 0.7 hedef

### E. Eval framework (Task 6)
- 87 sample × 4 kondisyon (A baseline, B RAG, C math, D hybrid)
- ROUGE + BERTScore + recall@k
- Subject breakdown plot (matplotlib)

---

## 4. Karakter Notları (P1 + P2 + P3'te doğrulandı)

- **Spec'i körü körüne uygulama** — P3'te 35+ sapma kayıtlı; P4'te devam
- **Empirical kanıt** — Sapma 39 dersi: varsayım yerine veri-driven (F-1 benchmark P3 örneği)
- **Doğru aleti seç** — RAG bilgi için, agent+tool reasoning için (P4 math agent doğal uygulaması)
- **Top-1 ≠ kalite** — recall@k kanıt için
- **Smoke test maliyeti düşük** — her component (PDF parser, sympy) önce smoke
- **YAGNI** — EBA/Khan eklemeyiz; Process.hierarchical eklemeyiz; scope kontrol
- **CI gate** — `ruff check + ruff format --check + pytest` commit öncesi şart
- **Sapma yaz** — `docs/p4/IMPLEMENTATION_NOTES.md` her bilinçli ayrılan karar için
- **Türkçe yorum** — kod açıklayıcı, WHY anlatır WHAT değil

---

## 5. P4'ün Açık Soruları (Task öncesi konuş)

1. **PDF kaynağı**: Kullanıcı 5 MEB PDF temin edecek mi yoksa placeholder ile mi başlayalım? Hangi dersler ilk pas?
2. **ÖSYM JSON formatı**: Manuel mi parse edeceğiz, resmi açık veri var mı?
3. **Math agent backend**: LangGraph subgraph mı CrewAI Crew mı? (LangGraph daha hafif, sympy tool için yeter)
4. **Recall@k gold etiketleme**: Manuel mi (30 dk iş) yoksa keyword auto-match mi?
5. **Eval cost**: 87×4 = 348 LLM call ~$1-2 — bütçede mi?

---

## 6. Faz Roadmap (P3 → P7)

P3'te kullanıcıyla beraber tasarlandı (2026-05-04 brainstorming):

```
P4 — Content & Tools           ← şu an, fresh session
P5 — Backend Production         (Postgres + Auth + Rate limit + KVKK + vLLM)
P6 — Frontend & UX              (Next.js + chat UI + KaTeX + PWA)
P7 — Infrastructure (eski P4)  (K8s + EKS + Terraform — taşındı `docs/p7/`)
```

Eski `docs/p4/GUIDE.md` → `docs/p7/GUIDE.md` rename edildi (2026-05-04).

---

## 7. Cost Bilinci — vLLM ne zaman?

P5'te self-hosted vLLM Qwen3 etkin olacak. AWS g4dn.xlarge spot ~$115/ay.

| Senaryo | Aylık |
|---|---|
| Anthropic-only (10k req/ay) | ~$50 |
| Kendi GPU (24/7) | ~$115 |
| Hybrid | ~$155 |

İlk 1000 user altında Anthropic ucuz; scale arttıkça kendi GPU kazanır.
KVKK için kendi GPU avantaj (veri dışarı çıkmaz).

P4'te dev'de yine **Anthropic API** kullanılır. vLLM P5 işi.

---

## 8. P4 Kickoff Prompt Taslağı

Aşağıdaki metni **fresh Claude Code session'a ilk mesaj olarak** ver. P3
kickoff pattern'ini takip ediyor + P3'te öğrenilenler eklendi.

```markdown
> Ben senior seviyeye doğru ilerleyen bir Python backend geliştiricisiyim.
> Türkçe konuşuyorum, kod yorumlarını Türkçe ve açıklayıcı bekliyorum.
>
> **EduAI Platform** — Türkçe lise eğitimi için AI destekli soru-cevap
> platformu. P1 (FastAPI 2026-04-18), P2 (Qwen3-4B+LoRA 2026-04-28),
> P3 (RAG+LangGraph+CrewAI 2026-05-03) finalize. Şimdi P4 (Content & Tools)
> başlıyoruz. Kod GitHub'da: `yasirbeydiligit/eduai-platform`.
>
> Lütfen şu sırayla oku:
> 1. `docs/p3/P4_HANDOFF.md` — bu dosya, P3'ten P4'e geçiş bağlamı
> 2. `docs/p4/CONCEPT.md` — P4'ün kavramsal arka planı (Sapma 39 dersi merkez)
> 3. `docs/p4/SPEC.md` — implementation spec
> 4. `docs/p4/TASKS.md` — adım adım görev listesi
>
> P3 sapmaları için referans: `docs/p3/IMPLEMENTATION_NOTES_ARCHIVED.md`
> (35+ sapma, Sapma 39 P4'ün varlık nedeni).
>
> **Karakter notları (P1+P2+P3'te doğrulandı):**
> - Spec'i körü körüne uygulama — sapma yaz, gerekçeyle
> - Empirical kanıt — varsayım yerine veri-driven (Sapma 39 örneği)
> - Doğru aleti seç — RAG bilgi için, sympy reasoning için (P4 math agent)
> - Top-1 ≠ kalite — recall@k kanıt için
> - Smoke test maliyeti düşük
> - YAGNI — EBA/Khan eklemiyoruz, scope kontrol
> - CI gate disiplini: ruff check + format + pytest commit öncesi
>
> **P3 final stack:**
> - Embedding: intfloat/multilingual-e5-large (1024-dim, e5 prefix)
> - Vector DB: Qdrant v1.12.4
> - LLM dev: claude-haiku-4-5 (Anthropic API)
> - Pipeline: LangGraph + CrewAI 1.14
> - Test: in-memory Qdrant + FakeEmbedder + MockLLM, 22 PASSED
>
> **P4 hedefleri (Sapma 39 fix):**
> - 5+ MEB PDF + 100+ ÖSYM soru ingestion (gerçek corpus)
> - Math tool agent (sympy + LLM + verification)
> - Validator weak indicator listesi genişletme
> - Recall@k metric implementation
> - 87×4 eval framework: D_hybrid Δ pozitif 80%+ subject
>
> **Task 0'dan başla, ihtiyacın olan kararları konuşalım, sonra uygulayalım.**
>
> Memory: `project_eduai.md`, `user_role.md`, `feedback_code_style.md`,
> `reference_p1_implementation_notes.md`, `reference_p2_implementation_notes.md`,
> `reference_p3_implementation_notes.md` zaten yüklü.
```

---

## 9. Pre-flight Checklist (P4 başlamadan önce kullanıcı yapsın)

```
[ ] git status temiz (P3 finalize commit edilmiş)
[ ] docker-compose up qdrant -d → /readyz 200 OK
[ ] source .venv-agents/bin/activate
[ ] PYTHONPATH=. pytest agents/tests/ → 22 PASSED
[ ] ml/.env içinde ANTHROPIC_API_KEY var
[ ] (opsiyonel) 5 MEB PDF agents/data/pdfs/ için hazır
[ ] (opsiyonel) ÖSYM JSON sample agents/data/osym/ için hazır
```

---

**Hand-off tamamlandı.** P3'ün gerçek bilim insanı disipliniyle (Sapma 39'u
gizlemek yerine ışığa çıkardık) P4'e geçiş hazır. Şimdi gerçek içerikle
**ürünleştirme** başlıyor.
