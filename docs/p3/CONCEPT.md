# P3 — Akıllı Ajan: Konsept Kılavuzu

> P2'den model çıktı. P1'den API ayakta. Şimdi bunları birleştirip "düşünen" bir sistem yapıyoruz.

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

## Başlamadan önce sorular

1. RAG ile fine-tuning'in farkı ne? Hangisi ne zaman kullanılır?
2. Chunk size 500 mi 1000 mi olmalı? Tradeoff nedir?
3. LangGraph'ta "state" neden önemli?
4. CrewAI'da agent ve task farkı nedir?
5. Vektör similarity search nasıl çalışır?
