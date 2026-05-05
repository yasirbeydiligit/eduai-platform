# P6 — Görev Listesi: Frontend & UX

> P5 FINALIZED. Backend production hazır. Transfer: `docs/p5/P6_HANDOFF.md`.

---

## Task 0 — Next.js + shadcn ⏱ ~45 dk
```
1. npx create-next-app@latest web --ts --tailwind --app
2. npx shadcn init
3. shadcn add: button, input, card, dialog, dropdown, toast, scroll-area
4. lib/api/types.ts: openapi-typescript ile P5 types
5. .env.local: NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Task 1 — Auth flow ⏱ ~2 saat
```
1. app/(auth)/login + signup + react-hook-form + zod
2. lib/auth/cookies.ts: httpOnly JWT (Next.js Server Actions)
3. KVKK consent — signup zorunlu checkbox
4. middleware.ts: korumalı route'lar (/chat → /login)
5. Smoke: signup → login → /chat
```

## Task 2 — Chat UI core ⏱ ~3 saat
```
1. components/chat/{MessageList, MessageBubble, InputBar}
2. lib/store/chat.ts: Zustand
3. react-markdown + remark-gfm + rehype-highlight
4. KaTeX (react-katex)
5. Smoke: 5 mesajlık konuşma render + scroll
```

## Task 3 — Streaming (SSE) ⏱ ~1.5 saat
```
1. lib/api/stream.ts: EventSource wrapper
2. InputBar submit → streaming state
3. Status indicator: "Aranıyor / Cevap / Doğrulanıyor"
4. Token-by-token render
5. Smoke: 1 soru → progress + final
```

## Task 4 — Math + Source render ⏱ ~1 saat
```
1. MathBlock.tsx: $$...$$ → KaTeX
2. SourceCitation.tsx: chunk metadata tooltip
3. DisclaimerBanner.tsx: isMathAnswer ise üstte
4. Smoke: math cevap test
```

## Task 5 — History + Settings ⏱ ~1.5 saat
```
1. app/chat/[sessionId]: history fetch + render
2. Sidebar: session list (last 20)
3. app/settings: profile + KVKK actions
4. Account delete: confirm modal + cascade UI
5. Data export: JSON download
```

## Task 6 — Onboarding ⏱ ~45 dk
```
1. components/onboarding/Tour.tsx: 3 step modal
2. localStorage flag — tek seferlik
3. İçerik: hoş geldin → ilk soru → disclaimer
4. Smoke: ilk login → tour görünür
```

## Task 7 — Mobile + PWA ⏱ ~1 saat
```
1. Tailwind responsive: sm/md/lg test
2. public/manifest.json (PWA)
3. service-worker.js (opsiyonel offline)
4. iOS Safari + Android Chrome smoke
```

## Task 8 — Testler + finalize ⏱ ~2 saat
```
1. e2e/critical-path.spec.ts: signup → ask → history (Playwright)
2. components/__tests__/: 10+ component test (Vitest)
3. Lighthouse audit (90+ all)
4. README: stack, dev, build, deploy
```

---

## P6 tamamlandı mı?

```
[ ] Login/signup + KVKK consent
[ ] Chat markdown + KaTeX
[ ] SSE streaming
[ ] Math disclaimer banner
[ ] Session history + settings
[ ] Mobile responsive
[ ] Playwright E2E PASSED
[ ] Lighthouse 90+
[ ] docs/p6/P7_HANDOFF.md (Infrastructure — eski P4)
```
