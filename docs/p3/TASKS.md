# P3 — Görev Listesi: Akıllı Ajan

> P1 ayakta olmalı. P2 modeli hazır (yoksa OpenAI API ile mock başla).

---

## Task 0 — Yapı ve Qdrant ⏱ ~20 dk

### Claude Code'a ver:
```
Proje: eduai-platform/agents/

Görev: Klasör yapısını oluştur ve Qdrant bağlantısını test et.

1. SPEC.md'deki klasör yapısını oluştur
2. requirements.txt oluştur
3. docker-compose.yml'e Qdrant servisini ekle (P1'in docker-compose.yml'ini güncelle)
4. agents/test_connection.py: Qdrant'a bağlan, collection oluştur, test verisi yükle, sil
```

### Bunu kendin yap:
```bash
docker-compose up qdrant -d
python agents/test_connection.py
# "Qdrant bağlantısı başarılı" çıktısını gör
git add . && git commit -m "feat: P3 agents structure + Qdrant setup"
```

---

## Task 1 — RAG: indexer ⏱ ~45 dk

**Bu task'ta öğreniyorsun:** Text splitting, embeddings, vector DB, metadata.

### Claude Code'a ver:
```
Proje: eduai-platform/agents/

Görev: rag/embeddings.py ve rag/indexer.py yaz. SPEC.md'deki sınıfları tam uygula.

Ekstra gereksinimler:
- Indexer, aynı dökümanı iki kez yüklemeyi engelle (document_id kontrolü)
- Her chunk'ın metadata'sında: source (dosya adı), page_num, subject, chunk_index
- İlerleme göster: "Chunk 1/24 yüklendi..."
- Hata durumunda: hangi chunk hata verdi, neden

Test için: data/ klasörüne küçük bir .txt dosyası koy (10-20 cümle tarih konusu)
Script sonu: o dosyayı indexle, "X chunk yüklendi" yaz
```

### Bunu kendin yap:
```bash
python agents/rag/indexer.py
# Qdrant'a gidip collection'ı kontrol et
# http://localhost:6333/dashboard (Qdrant UI)
git add . && git commit -m "feat: document indexer with Qdrant"
```

---

## Task 2 — RAG: retriever ⏱ ~30 dk

### Claude Code'a ver:
```
Proje: eduai-platform/agents/

Görev: rag/retriever.py yaz. SPEC.md'deki EduRetriever sınıfını tam uygula.

Test scripti ekle (agents/test_retrieval.py):
1. Bir soru yaz: "Tanzimat Fermanı ne zaman çıktı?"
2. retrieve() çağır (k=4)
3. Her sonuç için yazdır:
   - Score (similarity)
   - İlk 100 karakter
   - Source dosya
4. get_context_string() çıktısını yazdır
```

### Bunu kendin yap:
```bash
python agents/test_retrieval.py
# Gelen chunk'lar soruyla alakalı mı?
# Score'lar mantıklı mı? (>0.7 iyi)
git add . && git commit -m "feat: RAG retriever with similarity search"
```

---

## Task 3 — LangGraph pipeline ⏱ ~1 saat

**Bu task'ta öğreniyorsun:** State machine, conditional edges, retry logic.

### Claude Code'a ver:
```
Proje: eduai-platform/agents/

Görev: graph/ klasörünü tamamla. SPEC.md'deki state.py, nodes.py, edges.py, pipeline.py yaz.

LLM için: P2 modeli henüz servis edilmiyorsa, generate_node'da OpenAI ya da Anthropic API kullan (mock olarak).
Model servisi için: transformers pipeline ile local model da kullanılabilir.

Ekstra:
- Her node girişinde ve çıkışında structlog ile log at
- pipeline.py sonuna test kodu ekle:
  state = {"question": "Tanzimat Fermanı nedir?", "subject": "tarih", "grade_level": 9, 
           "session_id": "test-123", "attempts": 0, "needs_retry": False}
  result = await pipeline.ainvoke(state)
  print(result["answer"])
  print(result["sources"])
```

### Bunu kendin yap:
```bash
python agents/graph/pipeline.py
# Cevap geldi mi? Sources dolu mu?
# needs_retry=True durumunu test et: çok kısa cevap üret
git add . && git commit -m "feat: LangGraph pipeline with retrieve-generate-validate"
```

---

## Task 4 — CrewAI entegrasyonu ⏱ ~45 dk

### Claude Code'a ver:
```
Proje: eduai-platform/agents/

Görev: crew/ klasörünü tamamla. SPEC.md'deki agents.py, tasks.py, tools.py yaz.

RAG Tool tanımı:
@tool
def search_education_materials(query: str, subject: str) -> str:
    """Eğitim materyallerinde arama yapar."""
    retriever = EduRetriever()
    docs = retriever.retrieve(query, subject)
    return retriever.get_context_string(docs)

Crew çalıştırma testi (crew/test_crew.py):
- Karmaşık soru: "Newton'un hareket yasaları ve Osmanlı'nın modernleşme süreci arasında 
  benzer bir dinamik var mı? Açıkla." (çok disiplinli soru)
- Researcher araştırsın, Writer yazsın
- Sonucu yazdır
```

### Bunu kendin yap:
```bash
python agents/crew/test_crew.py
# İki agent sırayla çalıştı mı?
# Final cevap coherent mi?
git add . && git commit -m "feat: CrewAI multi-agent for complex questions"
```

---

## Task 5 — P1 API güncelle ⏱ ~30 dk

**Bu task'ta öğreniyorsun:** Mevcut sisteme yeni feature ekleme.

### Claude Code'a ver:
```
Proje: eduai-platform/services/api/

Görev: P1 API'sine iki yeni şey ekle.

1. /v1/documents/upload endpoint'ini gerçekten Qdrant'a yükleyecek şekilde güncelle:
   - Upload sonrası DocumentIndexer.index_file() çağır
   - Response'a "chunks_indexed" sayısını ekle

2. /v1/questions/ask/v2 endpoint'i ekle:
   - LangGraph pipeline'ı çağır
   - AgentState'i oluştur
   - Pipeline çalıştır
   - QuestionResponse formatında döndür

3. dependencies.py'e:
   get_agent_pipeline() → compiled LangGraph
   get_document_indexer() → DocumentIndexer

Dikkat: ajan kodları agents/ klasöründe. API onları import edecek.
Monorepo yapısı için PYTHONPATH'e dikkat et.
```

### Bunu kendin yap:
```bash
docker-compose up
# Swagger UI: http://localhost:8000/docs
# /v1/questions/ask/v2 endpoint'i görünüyor mu?
# Döküman yükle → soru sor → kaynaklı cevap gel mi?
curl -X POST http://localhost:8000/v1/questions/ask/v2 \
  -H "Content-Type: application/json" \
  -d '{"question":"Tanzimat nedir?","session_id":"...","subject":"tarih","grade_level":9}'
git add . && git commit -m "feat: integrate LangGraph pipeline into FastAPI"
```

---

## Task 6 — Testler ⏱ ~30 dk

### Claude Code'a ver:
```
Proje: eduai-platform/agents/

Görev: tests/ klasörüne testler yaz.

tests/test_rag.py:
1. test_index_and_retrieve: küçük metin yükle → retrieve et → sonuç var mı?
2. test_retrieve_with_subject_filter: subject filtresi çalışıyor mu?
3. test_empty_retrieve: boş collection → boş liste döner mi?

tests/test_pipeline.py:
1. test_full_pipeline: soru → cevap + sources döner
2. test_retry_logic: kısa cevap → retry tetikleniyor mu?
3. test_pipeline_max_attempts: 3 denemede zorla bitiyor mu?

Mock kullan: gerçek Qdrant yerine in-memory mock.
```

### Bunu kendin yap:
```bash
pytest agents/tests/ -v
git add . && git commit -m "test: RAG and pipeline integration tests"
```

---

## P3 tamamlandı mı? Kontrol listesi

```
[ ] docker-compose up → API + Qdrant çalışıyor
[ ] Döküman yükle → Qdrant'ta görünüyor (http://localhost:6333/dashboard)
[ ] /ask/v2 → gerçek RAG cevabı, sources dolu
[ ] LangGraph retry: kötü cevap → otomatik tekrar deneme
[ ] CrewAI: karmaşık sorularda 2 agent çalışıyor
[ ] pytest agents/tests/ → geçiyor
[ ] README: tam kullanım kılavuzu
```
