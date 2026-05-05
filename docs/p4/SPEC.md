# P4 — Content & Tools: Proje Spesifikasyonu

---

## Proje özeti

**Ne:** P3 RAG mimarisini gerçek içerikle (MEB PDF + ÖSYM) besle, matematik
için sympy tool agent ekle, eval Δ pozitif yap.

**Klasör:** `eduai-platform/agents/` (P3'ten devam) + yeni alt-dizinler.

**Bağımlılık:** P3 FINALIZED — agents/, services/api, docker-compose Qdrant
çalışıyor olmalı.

---

## Klasör yapısı (yeni eklemeler)

```
agents/
├── content/                    ← YENİ: PDF + ÖSYM ingestion
│   ├── __init__.py
│   ├── pdf_parser.py           ← MEB PDF → structured text
│   ├── osym_loader.py          ← ÖSYM JSON → Qdrant
│   └── corpus_manager.py       ← multi-corpus orchestration
├── tools/                      ← YENİ: Math tool stack
│   ├── __init__.py
│   ├── sympy_tool.py           ← Symbolic math executor
│   ├── math_agent.py           ← sympy + LLM + verification
│   └── router.py               ← Soru classifier (math vs theory)
├── eval/                       ← YENİ: Eval framework
│   ├── __init__.py
│   ├── recall_at_k.py          ← Retrieval recall metric
│   ├── eval_full.py            ← 87 sample × 4 kondisyon
│   └── plots.py                ← matplotlib subject breakdown
├── data/
│   ├── pdfs/                   ← MEB PDF (.gitignored)
│   ├── osym/                   ← ÖSYM JSON arşivi
│   └── eval_gold_chunks.csv    ← Recall@k manuel etiketler
├── rag/, graph/, crew/, tests/  ← P3'ten devam
```

---

## Bileşen spesifikasyonları

### content/pdf_parser.py

```python
"""MEB PDF → structured chunks. pdfplumber backend (pypdf'den iyi)."""

class PDFParser:
    def parse(self, pdf_path: Path) -> list[Chunk]:
        """
        1. pdfplumber ile sayfa-sayfa extract
        2. Boş/header/footer satır temizle (regex)
        3. RecursiveCharacterTextSplitter (P3 indexer pattern)
        4. Her chunk: text, page_num, source_filename
        """

    def detect_quality(self, text: str) -> Literal["good", "degraded"]:
        """text/page < 60% → degraded (image-heavy, math/fizik). Skip + log."""
```

**Hedef dersler:** tarih, edebiyat, coğrafya, din, felsefe, biyoloji.
**Skip:** matematik, fizik, kimya (math agent yolu).

### content/osym_loader.py

```python
"""ÖSYM geçmiş yıl → Qdrant `osym_questions` collection."""

class OSYMLoader:
    def load(self, json_path: Path) -> int:
        """
        Format: {year, exam_type, subject, question, choices, answer_key}
        Her soru tek chunk:
            text = f"Soru: {question}\n\n{choices}"
            payload = {answer_key, year, exam_type, subject, ...}
        Vector size embedder ile aynı (1024).
        """
```

### content/corpus_manager.py

```python
"""Multi-corpus retriever — theory + osym ayrı collection.

EduRetriever extension: retrieve(query, corpus="theory"|"osym"|"both").
"both" → paralel retrieve + score normalize + merge.
"""
```

### tools/sympy_tool.py

```python
"""Symbolic math wrapper — sympy + numpy."""

@tool("solve_equation")
def solve_equation(equation: str) -> dict:
    """LLM-callable.
    
    Args: equation = "x**2 + 5*x + 6 = 0" (sympy syntax)
    Returns: {"roots": [-2, -3], "steps": [...], "error": None}
             veya {"error": "Çözülemedi: ..."}
    """

@tool("compute_derivative")
def compute_derivative(expression: str, variable: str = "x") -> dict: ...

@tool("compute_integral")
def compute_integral(expression: str, variable: str, lower=None, upper=None) -> dict: ...
```

### tools/math_agent.py

```python
"""LangGraph subgraph — math soru çözme akışı.

Akış: parse_question → sympy_tool → llm_explain → verify → format
"""

class MathAgent:
    async def solve(self, question: str, grade: int) -> dict:
        """
        1. parse: "x²+5x+6=0 çöz" → "x**2+5*x+6=0"
        2. sympy: roots = [-2, -3]
        3. llm_explain: adım adım Türkçe açıklama (sympy çıktısına ground'lu)
        4. verify: 2nd LLM "Bu çözüm matematik olarak doğru mu?"
        5. format: disclaimer + adım adım çıktı
        """
```

### tools/router.py

```python
"""Soru classifier — heuristic, LLM gereksiz."""

def classify(question: str) -> Literal["math", "theory"]:
    """Pattern: \\d, =, ²/³, türev/integral/denklem keyword → math.
    Belirsizlik → theory (default — RAG hata az hallucinate)."""
```

### eval/recall_at_k.py

```python
"""Retrieval recall — gold chunk top-k'de mi?"""

def compute_recall_at_k(
    retriever, eval_data, gold_chunks, k_values=[1, 4, 10]
) -> dict:
    """Her eval Q için retriever.retrieve() → gold top-k'de mi?
    Return: {1: 0.x, 4: 0.x, 10: 0.x}
    """
```

### eval/eval_full.py

```python
"""87 sample × 4 kondisyon eval — eval_ab.py genişletmesi."""

CONDITIONS = ["A_baseline", "B_rag", "C_math", "D_hybrid"]

# D_hybrid: router classify → math ise C, theory ise B
async def main():
    """Tüm 87 eval × 4 kondisyon = 348 LLM call (~$1-2 cost).
    ROUGE + BERTScore + recall@k (B/D) + subject breakdown."""
```

---

## requirements.txt eklemeleri

```
# P4 ek
sympy>=1.13
matplotlib>=3.9
pdfplumber>=0.11        # pypdf'den iyi tablo extraction
```

---

## API güncellemesi

`/v1/questions/ask/v2` router classify ekle:
```python
classification = router.classify(request.question)
if classification == "math":
    response = await math_agent.solve(question, grade_level)
else:
    response = await pipeline.ainvoke(state)  # mevcut RAG yolu
```

ENV `MATH_AGENT_ENABLED=true|false` (eval karşılaştırma için).

---

## Teslim kriterleri

- [ ] 5+ MEB konu PDF'i indeksli (`agents/data/pdfs/` → `eduai_documents`)
- [ ] 100+ ÖSYM sorusu (`osym_questions` collection)
- [ ] Math agent: 3 örnek (denklem, türev, integral) doğru cevap
- [ ] Math agent verification: tutarsız çözüm flag
- [ ] Recall@k: 30 manuel etiketli sample üzerinde, recall@4 ≥ 0.7
- [ ] Eval full (87×4): D_hybrid Δ pozitif 80%+ subject'te
- [ ] Validator weak indicator listesi 10+ pattern
- [ ] `pytest agents/tests/` 35+ PASSED
- [ ] README güncel: PDF upload, math agent, eval çalıştırma
