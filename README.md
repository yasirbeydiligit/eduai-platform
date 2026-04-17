# EduAI Platform — Öğrenme Rehberi

Türkçe lise eğitimi için AI-powered soru-cevap platformu.
13 haftalık öğrenme yolculuğu: FastAPI → Fine-tuning → Agents → Cloud.

---

## Klasör yapısı

```
eduai-platform/
├── docs/                    ← Öğrenme kılavuzları (buradan başla)
│   ├── p1/
│   │   ├── CONCEPT.md       ← Önce oku: FastAPI, Docker, CI kavramları
│   │   ├── SPEC.md          ← Claude Code'a ver: mimari kararlar
│   │   └── TASKS.md         ← Günlük çalışma: adım adım görevler
│   ├── p2/
│   │   ├── CONCEPT.md       ← PyTorch, LoRA, QLoRA, MLflow kavramları
│   │   ├── SPEC.md          ← Fine-tuning pipeline spesifikasyonu
│   │   └── TASKS.md         ← Eğitim görevleri
│   ├── p3/
│   │   ├── CONCEPT.md       ← RAG, LangGraph, CrewAI kavramları
│   │   ├── SPEC.md          ← Agent sistemi spesifikasyonu
│   │   └── TASKS.md         ← Ajan geliştirme görevleri
│   └── p4/
│       └── GUIDE.md         ← Kubernetes, AWS, Terraform, monitoring
│
├── services/api/            ← P1 kodu: FastAPI backend
├── ml/                      ← P2 kodu: Fine-tuning pipeline
├── agents/                  ← P3 kodu: RAG + LangGraph + CrewAI
└── infra/                   ← P4 kodu: Terraform + K8s + Helm
```

---

## Nasıl kullanırsın?

Her proje için sıra: **CONCEPT → SPEC → TASKS**

1. `CONCEPT.md` → Kavramları anla. Kör kopyalamak için değil.
2. `SPEC.md` → Mimari kararlar. Claude Code'a master prompt olarak ver.
3. `TASKS.md` → Her task = bir Claude Code session. Bitince `git commit`.

---

## Zaman çizelgesi

| Proje | Süre | İçerik |
|-------|------|--------|
| P1 | 0–3 hafta | FastAPI · Docker · GitHub Actions |
| P2 | 3–7 hafta | PyTorch · LoRA · MLflow |
| P3 | 5–9 hafta | LangChain · RAG · LangGraph · CrewAI |
| P4 | 9–13 hafta | Kubernetes · AWS EKS · Terraform · Monitoring |

P2 ve P3 paralel başlayabilir. P4 için P1+P2+P3 tamamlanmış olmalı.

---

## Hızlı başlangıç (P1 için)

```bash
git clone https://github.com/kullaniciadi/eduai-platform
cd eduai-platform
docker-compose up
# http://localhost:8000/docs
```
