# P6 — Frontend & UX: Spesifikasyon

**Stack:** Next.js 15 + TypeScript + shadcn/ui + Tailwind + Zustand + react-markdown + KaTeX.

**Klasör:** `web/` (yeni top-level dizin).

---

## Klasör yapısı

```
web/
├── app/
│   ├── (auth)/{login,signup}/page.tsx
│   ├── chat/page.tsx
│   ├── chat/[sessionId]/page.tsx
│   ├── settings/page.tsx
│   └── layout.tsx
├── components/
│   ├── chat/{MessageList,MessageBubble,InputBar}.tsx
│   ├── chat/{MathBlock,CodeBlock,SourceCitation,DisclaimerBanner}.tsx
│   ├── auth/{LoginForm,SignupForm,KvkkConsent}.tsx
│   ├── onboarding/Tour.tsx
│   └── ui/                      ← shadcn primitives
├── lib/
│   ├── api/{client,types,stream}.ts
│   ├── auth/cookies.ts
│   └── store/chat.ts            ← Zustand
├── public/icons/                ← PWA
└── e2e/                         ← Playwright
```

---

## Ana bileşenler

### MessageBubble.tsx
```tsx
interface MessageBubbleProps {
  role: 'user' | 'assistant'
  content: string
  sources?: string[]
  confidence?: number
  isMathAnswer?: boolean
}
// markdown + KaTeX + source citations + (math ise) disclaimer
```

### Streaming (P3 SSE)
```typescript
const eventSource = new EventSource('/api/proxy/ask/v2/stream', {...})
eventSource.addEventListener('node_completed', (e) => {
  // Status: "Aranıyor...", "Cevap üretiliyor..."
})
eventSource.addEventListener('final', (e) => {
  // Final render
  eventSource.close()
})
```

### DisclaimerBanner.tsx
```tsx
<div className="bg-yellow-50 border-l-4 border-yellow-400 p-4">
  <strong>⚠ AI Çözümü</strong>
  Bu çözüm yapay zeka tarafından üretilmiştir. Adımları kontrol et,
  öğretmenden teyit al.
</div>
```

---

## API integration

P5 OpenAPI'den TypeScript tipleri otomatik:
```bash
openapi-typescript http://api/openapi.json -o lib/api/types.ts
```

Type-safe wrapper:
```typescript
export const api = {
  ask: (req: QuestionRequest) => fetch('/api/v1/questions/ask/v2', {...}),
  history: (sid: string) => fetch(`/api/v1/sessions/${sid}/messages`),
}
```

---

## Teslim kriterleri

- [ ] Login/signup + KVKK consent (zorunlu checkbox)
- [ ] Chat markdown + KaTeX render
- [ ] SSE streaming — node-by-node status
- [ ] Math cevaplarda disclaimer banner
- [ ] Session history sidebar
- [ ] Settings: data export + account delete
- [ ] Mobile responsive
- [ ] PWA manifest (opsiyonel)
- [ ] Playwright E2E: signup → ask → history (3 critical path)
- [ ] Vitest 10+ component testleri
- [ ] Lighthouse 90+ skoru
