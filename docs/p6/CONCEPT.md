# P6 — Frontend & UX: Kavramsal Kılavuz

> P5 backend production-ready. P6 **kullanıcı yüzü**: web chat UI,
> disclaimer banner'ları, onboarding, conversation history. Ürünleştirmenin
> son katmanı.

---

## P6'nın varlık nedeni

P3-P5 sonu API tam, secure, scalable. Ama:
- ❌ Sadece curl/Postman ile kullanılıyor (öğrenci kullanmaz)
- ❌ Math agent disclaimer'ı UI'da yok (sorumluluk)
- ❌ Conversation history kullanıcıya görünmüyor
- ❌ Onboarding yok

P6 bu 4 sınırı çözer + **mobil opsiyonel** (kullanıcının "plus" demesi).

---

## Büyük resim

```
P5 Backend (production)            P6 Frontend & UX
─────────────────────              ────────────────
JWT auth API                       ✅ Login/signup formları
/ask/v2 endpoint                   ✅ Chat UI (markdown render)
/ask/v2/stream (SSE)               ✅ Streaming with status indicators
session_id manuel                  ✅ Session list + history
disclaimer text                    ✅ Banner + tooltip "AI çözümü"
                                   ✅ Onboarding (3 step tour)
                                   ✅ Math render (KaTeX)
                                   ✅ (opsiyonel) PWA mobil
```

---

## Stack seçimi

**Önerim: Next.js 15 + TypeScript + shadcn/ui + Tailwind + Zustand**

Sebep:
- Next.js 15 RSC + Server Actions → BFF pattern (CORS yok)
- TypeScript zorunlu (P5 API contract paylaşımı)
- shadcn/ui Türkçe i18n + accessibility yerleşik
- Tailwind utility-first hız
- Zustand chat state (Context'e göre lighter)

**Alternatif:** HTMX + Jinja2 (FastAPI native, daha az JS) — daha basit ama
streaming/animasyonlu chat UI sınırlı.

---

## Chat UI bileşenleri

- **MessageList** — virtualized scroll
- **MessageBubble** — markdown (`react-markdown` + `remark-gfm`)
- **CodeBlock** — syntax highlighting (`shiki`)
- **MathBlock** — KaTeX (`react-katex`) — math agent çıktısı için
- **SourceCitation** — chunk metadata gösterimi (P3 sources)
- **DisclaimerBanner** — math cevaplarının üstünde sarı uyarı

---

## Cevap tipleri için UI

| Cevap tipi | UI render |
|---|---|
| Theory (RAG) | Markdown + source citations + confidence skoru |
| Math agent | KaTeX render + sympy steps + **sarı disclaimer** |
| Hybrid | İçeriğe göre otomatik |
| Streaming (P3 SSE) | Token typing animasyonu |

---

## KVKK + güvenlik

- Cookie-based JWT (httpOnly, secure, samesite=strict)
- CSRF token (Next.js built-in)
- KVKK consent screen — signup ilk adım
- Account settings: data export + delete

---

## Onboarding flow

1. Hoş geldin: "Lise dersleri için AI asistanın"
2. İlk soru: "Konu seç → soru sor"
3. Disclaimer kabul: "AI cevapları kontrol et, son söz öğretmen"

LocalStorage flag → tek seferlik.

---

## Mobile responsiveness

- Tailwind breakpoint (sm/md/lg)
- Mobile-first: chat UI default mobile, desktop side panel
- (opsiyonel) PWA manifest → "Ana ekrana ekle"
