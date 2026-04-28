# P3 — Akıllı Ajan: Konsept Kılavuzu

> P2'den model çıktı. P1'den API ayakta. Şimdi bunları birleştirip "düşünen" bir sistem yapıyoruz.
>
> **📥 P2'den gelen (2026-04-28 itibariyle):**
> - **Base model:** `Qwen/Qwen3-4B-Instruct-2507` (Apache 2.0; Phi-3 Türkçe yetersizdi, Qwen3'e geçildi)
> - **LoRA adapter:** Drive `/content/drive/MyDrive/eduai_qwen3-4b-instruct-2507_ckpt/`
>   (P3'te HF Hub'a push edilebilir)
> - **Final eval metrikleri:** ROUGE-L 0.249, BERTScore F1 0.640, eval_loss 1.347
> - **Inference profili:** ~22-25 sn/cevap T4 (256 token); A100'de ~5 sn
> - **Karakter:** akıcı Türkçe + pedagojik ton + markdown yapı (bold başlık, liste);
>   bilgi doğruluğu hataları RAG context'i ile düzeltilecek
>
> **Tam transfer paketi:** [`docs/p2/P3_HANDOFF.md`](../p2/P3_HANDOFF.md) — adapter
> yükleme kod örneği, inference wrapper imzası, P2 sapmaları özet, kickoff prompt.

---

## Büyük resim: ne yapıyoruz?

Bir öğrenci soru sorduğunda sistem şunu yapıyor:

```
Soru geldi
    ↓
LangGraph: "Bu soru hangi türde?" (routing)
    ↓
Retriever: Ders materyallerinden ilgili bölümleri bul (RAG)
    ↓
Generator: Fine-tuned modelle cevap üret
    ↓
Validator: Cevap yeterliyse döndür, değilse tekrar dene
    ↓
Cevap + kaynaklar öğrenciye gönderilir
```

Bu bir **agentic pipeline** — her adım bir karar veriyor.

---

## Temel kavramlar

### 1. RAG nedir? Neden sadece fine-tuning yetmez?

Fine-tuned model, eğitim verilerini "öğrendi" ama güncel bilmiyor. Vizyon Koleji yeni bir ders materyali eklediğinde modeli yeniden eğitmek saatler sürer.

**RAG (Retrieval-Augmented Generation):**
1. Dökümanları vektörleştir, veritabanına kaydet (indexing)
2. Soru gelince: soruya benzer bölümleri bul (retrieval)
3. Bulunan bölümleri + soruyu modele ver (augmentation)
4. Model bu bağlamla cevap üretir (generation)

```
"Tanzimat nedir?" → vektör ara → tarih_9_unite3.pdf s.47 → modele ver → cevap
```

Sonuç: Model sürekli güncel kalmak zorunda değil. Yeni döküman yüklenince sadece indexleniyor.

---

### 2. Vektör DB nedir?

Metni direkt arama yapamayız (anahtar kelime eşleşmesi yüzeysel). Bunun yerine:
- Her metin parçasını (chunk) bir sayı vektörüne çeviririz
- Benzer anlamlı metinler → benzer vektörler
- "Tanzimat Fermanı" ile "1839 Osmanlı reformu" aynı bölgeye düşer

```python
# Embedding = metni vektöre çevirme
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("emrecan/bert-base-turkish-cased-mean-nli-stsb-tr")
vector = model.encode("Tanzimat Fermanı nedir?")
# vector: [0.23, -0.14, 0.87, ...] — 768 boyutlu sayı dizisi
```

**Qdrant:** Bu vektörleri saklar ve hızlı arama yapar. PostgreSQL'in vector extension'ından daha hızlı, production-ready.

> **📝 Türkçe embedding model seçimi (2026 reality check):**
> SPEC'te `emrecan/bert-base-turkish-cased-mean-nli-stsb-tr` (2021) öneriliyor —
> hâlâ çalışır ama daha güncel multilingual modeller (`BAAI/bge-m3`,
> `intfloat/multilingual-e5-large`, `Alibaba-NLP/gte-multilingual-base`) Türkçe'de
> belirgin daha iyi. P3 Task 1'de **2-3 modeli kıyaslamaya değer** — küçük benchmark
> (50 soru × 3 model × cosine sim ortalama) 30 dakikada karar verdirir.

---

### 3. Chunking nedir?

Bir PDF'i direkt modele veremezsin — çok uzun. Önce parçalara bölmek gerekir.

```python
# chunk_size=500: her parça 500 token
# chunk_overlap=50: parçalar 50 token örtüşür (bağlam kaybetmemek için)
chunks = text_splitter.split_text(document)
```

Kötü chunking → bağlamı kopuk parçalar → kötü cevaplar. Bu önemli.

---

### 4. LangChain nedir?

LangChain, farklı AI bileşenlerini (model, retriever, memory, tools) birbirine bağlayan framework. Şöyle düşün: lego parçaları. LangChain onları birleştiriyor.

```python
from langchain.chains import RetrievalQA

chain = RetrievalQA.from_chain_type(
    llm=your_model,
    retriever=vector_store.as_retriever(),
    return_source_documents=True,
)

result = chain.invoke({"query": "Tanzimat nedir?"})
# result["answer"] + result["source_documents"]
```

---

### 5. LangGraph nedir? LangChain'den farkı?

LangChain: A → B → C şeklinde doğrusal chain.

LangGraph: **State machine.** Agent'ın birden fazla durumu var ve kararlar duruma göre alınıyor.

```python
# State = agent'ın "hafızası"
class AgentState(TypedDict):
    question: str
    retrieved_docs: list
    answer: str
    attempts: int
    needs_retry: bool

# Düğümler = adımlar
graph.add_node("retrieve", retrieve_documents)
graph.add_node("generate", generate_answer)
graph.add_node("validate", validate_answer)

# Kenarlar = karar mantığı
graph.add_conditional_edges("validate", 
    lambda state: "retry" if state["needs_retry"] else "end")
```

Neden önemli? Eğer ilk cevap yetersizse sistem otomatik retry yapabiliyor. LangChain ile bu zor, LangGraph ile doğal.

---

### 6. CrewAI nedir?

Birden fazla agent'ı rollere göre organize eder.

```python
from crewai import Agent, Task, Crew

researcher = Agent(
    role="Eğitim Araştırmacısı",
    goal="Soruyla ilgili doğru kaynakları bul",
    backstory="Türk eğitim sisteminde uzman...",
    tools=[rag_tool],
)

writer = Agent(
    role="Pedagojik Yazar",
    goal="Öğrenci seviyesine uygun cevap yaz",
    backstory="Lise öğretmeni deneyimine sahip...",
)

task1 = Task(description="...", agent=researcher)
task2 = Task(description="...", agent=writer)

crew = Crew(agents=[researcher, writer], tasks=[task1, task2])
result = crew.kickoff(inputs={"question": "..."})
```

---

### 7. Bu projede LangGraph mı CrewAI mı?

**İkisini birlikte kullanıyoruz, farklı amaçlarla:**

- **LangGraph:** Ana pipeline akışı — retrieve → generate → validate döngüsü
- **CrewAI:** Karmaşık sorular için — "Bu soru birden fazla konuyu kapsıyor, önce tarih agent'ı araştırsın, sonra matematik agent'ı eklesin"

---

### 8. Memory nedir?

Bir öğrenci 10 soru sorduğunda sistem önceki soruları "hatırlamalı". Aksi halde her soru bağımsız işlenir ve "bir önceki sorumu açar mısın?" anlaşılmaz.

```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=5)  # son 5 mesajı tut
```

---

## P2'den taşınan dersler (P3'te tekrar etmeyelim)

P2 boyunca 31 bilinçli sapma kayıtlı; bunların **5'i P3'e doğrudan ışık tutuyor:**

1. **Loss düşüşü ≠ kalite iyileşmesi.** P2'de Phi-3 baseline'da loss 2.11→1.80
   "iyi" görünüyordu, üretim çöp. **P3'te de:** retrieval recall@k metriği
   yüksek olabilir ama retrieved chunk'lar gerçekten alakalı mı? Sample
   inspection eval döngüsünün parçası.

2. **Differential diagnosis pattern.** Bug çıktığında suspect'i izole et
   (önce retriever'ı kanıtla, sonra generator'ı). P2'de "fine-tuning bozdu"
   varsayımı yanlıştı — base model zaten bozuktu. P3'te de "RAG yanlış
   chunk getirdi" demeden önce retriever standalone test et.

3. **Spec uyarılarını küçümseme.** P2'de CONCEPT "Phi-3 Türkçe: Orta" diyordu,
   biz devam ettik, 2-3 iterasyon kaybettik. **P3'te dikkat:** embedding model,
   chunk strategy, validator threshold gibi parametrelerde "default ile başla"
   yerine erken benchmark.

4. **Smoke test maliyeti düşük, değeri yüksek.** Yeni component eklerken
   (Qdrant, LangGraph, CrewAI) önce minimal smoke test (3-5 dk). Full
   pipeline integration'a kadar bekleme.

5. **Doğru aleti tanı:** style/format için fine-tuning, bilgi doğruluğu için
   RAG. P2 "style provider" olarak iyi çalışıyor; P3'ün **görevi RAG ile
   bilgi doğruluğunu sağlamak.** Tersini yapmaya çalışma (örn. fine-tuning'i
   "daha iyi" hale getirmeyi denemek P2 sonu kararı zaten reddetti).

## P2 → P3 inference latency gerçeği

T4 Colab'da P2 inference: **~22-25 sn/cevap (256 token)**. P3'te RAG context
input'a eklenince (~1500 token) bu **30+ sn'ye çıkacak**. Production değil.

**P3 Task'larında ele alınmalı:**
- **Geliştirme döngüsü:** Mock LLM (Anthropic API ile fast testing) zaten
  TASKS.md Task 3'te öneriliyor — kullan. Asıl P2 modeli ile testleri sona bırak.
- **Production hazırlığı:** vLLM serving (~10x), speculative decoding (~30%),
  adapter merge + AWQ quantization. P3 sonu "production gateway" task'ında.

## Başlamadan önce sorular

1. RAG ile fine-tuning'in farkı ne? Hangisi ne zaman kullanılır?
2. Chunk size 500 mi 1000 mi olmalı? Tradeoff nedir?
3. LangGraph'ta "state" neden önemli?
4. CrewAI'da agent ve task farkı nedir?
5. Vektör similarity search nasıl çalışır?
6. P2 adapter'ı RAG context'i ile nasıl entegre olacak — system prompt mu, user prompt'a inline mı, retrieval-augmented chat template mi?
7. Validator naive length-based değil, kalite-based olabilir mi? (NLI model, LLM-as-judge)
