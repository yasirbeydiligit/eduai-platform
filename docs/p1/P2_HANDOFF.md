# P1 → P2 Hand-off

> Fresh Claude session P2'ye başlarken bu dosya + `docs/p2/CONCEPT.md` + `docs/p2/SPEC.md` + `docs/p2/TASKS.md` okunmalı.

---

## P1 durumu (as-of 2026-04-18)

### Tamamlandı
- **FastAPI backend** (`services/api/`): 5 core endpoint + `/health`, in-memory mock services
- **13 test yeşil** (pytest + TestClient), **ruff temiz** (check + format)
- **3-job GitHub Actions pipeline** (lint → test → docker-build) — `yasirbeydiligit/eduai-platform`
- **Multi-stage Dockerfile** (284MB, non-root `appuser`, curl HEALTHCHECK, `--workers 2`)
- **docker-compose** dev config (hot reload + healthcheck)
- **README İngilizce** (recruiter/senior dev ready)
- **Global exception handler** (SPEC kural 4) — P1 kapama sırasında eklendi

### Bilinçli kabul edilen eksiklikler
- **Docker image <200MB** hedefi karşılanmadı (284MB) — Alpine uyumsuzluğu + P2'de PyTorch için problematik; P4'te distroless/scratch denenecek
- `uvicorn[standard]` extras tam trim edilmedi (`watchfiles`, `websockets` etc.); prod için net perf farkı yok, diminishing return

### Nerede ne var
- `docs/p1/SPEC.md` → P1'in güncel ve **gerçek** durumu (inline 📝 Implementation note callout'ları ile)
- `docs/p1/IMPLEMENTATION_NOTES_ARCHIVED.md` → P1 boyunca alınan her sapma kararının gerekçesi (Task 0 → 7 + Task 5/6.5 mikro-düzeltmeler)

---

## P2 için önemli bağlam

### P1 kodu dokunulmaz
P2 bağımsız bir `ml/` dizininde gelişir. P1 FastAPI kodu P2'yi etkilemez. P2 çıktısı (fine-tuned model + adapter weights) P3'te P1 API'si arkasına takılacak — P1/P2 arası köprü P3'te kurulur.

### Tekrar eden kod pattern'leri (P2'de otomatik uygulanmalı)

| Pattern | Yanlış | Doğru |
|---------|--------|-------|
| UTC timestamp | `datetime.utcnow()` | `datetime.now(UTC)` |
| Pydantic mutable default | `Field(default=[])` | `Field(default_factory=list)` |
| FastAPI dependency (yeni endpoint yazarsan) | `service: T = Depends(fn)` | `service: Annotated[T, Depends(fn)]` |
| String enum | `class X(str, Enum):` | `class X(StrEnum):` |

P2'de PyTorch + transformers kullanacaksan bu pattern'ler daha az uygulanır (FastAPI yok), ama **Pydantic mutable default** kuralı dataset/config şemalarında hâlâ geçerli.

### User tercihleri (memory'de de kayıtlı)
- Türkçe iletişim + **Türkçe ve açıklayıcı kod yorumları**
- Spec'i körü körüne uygulama — mantıksız/eskimiş noktaları tespit edip **sapmayı gerekçesiyle açıkla**, sessizce düzeltme
- Her sapmayı `docs/p2/IMPLEMENTATION_NOTES.md`'de not et (P1 pattern'i tekrarlanır)
- Faz sonunda SPEC.md bu notlarla güncellenir + archive yapılır

### İş akışı şablonu (P1'de çalıştığı kanıtlandı)

```
Fresh session → CONCEPT.md oku → SPEC.md oku → TASKS.md'den sırayla görevler
Her task → çalışma sırasında sapmaları IMPLEMENTATION_NOTES.md'ye yaz
Faz sonu → SPEC.md'yi notlarla güncelle → NOTES.md'yi ARCHIVED'e rename → P(N+1)_HANDOFF.md yaz
```

### Araç durumu
- `pytest`, `ruff`, `docker`, `docker-compose` lokalde çalışıyor
- GitHub Actions push'ta otomatik (`lint → test → docker-build`)
- GitHub auth: HTTPS + Personal Access Token

---

## P2 kickoff — fresh Claude session'a söylenmesi gerekenler

Aşağıdaki metni kopyalayıp yeni Claude Code session'ına ilk mesaj olarak ver:

---

> Ben senior seviyeye doğru ilerleyen bir Python backend geliştiricisiyim. Türkçe konuşuyorum, kod yorumlarını Türkçe ve açıklayıcı bekliyorum.
>
> **EduAI Platform** adlı 4 fazlı AI eğitim platformu öğrenme projemin **P2 (fine-tuning pipeline)** fazına başlıyoruz. P1 (FastAPI backend) 2026-04-18'de tamamlandı. Kod GitHub'da: `yasirbeydiligit/eduai-platform`.
>
> Lütfen şu sırayla oku:
> 1. `docs/p1/P2_HANDOFF.md` — P1'den P2'ye geçiş bağlamı (bu dosya)
> 2. `docs/p2/CONCEPT.md` — P2'nin kavramsal arka planı
> 3. `docs/p2/SPEC.md` — P2'nin implementation spec'i
> 4. `docs/p2/TASKS.md` — adım adım görev listesi
>
> **Karakter notları:**
> - Spec'i körü körüne uygulama — mantıksız noktaları gerekçesiyle düzelt
> - Her sapmayı `docs/p2/IMPLEMENTATION_NOTES.md`'ye yaz (P1'deki gibi)
> - Task 0'dan başla, spec'i bana ver, sonra uygulayalım
>
> Memory'deki `project_eduai.md`, `user_role.md`, `feedback_code_style.md` ve `reference_p1_implementation_notes.md` dosyaları zaten yüklü olacak — onlardaki tercihlere ve pattern'lere uygun davran.

---

## Notlar

- `docs/p2/` dizini zaten hazır (CONCEPT/SPEC/TASKS); bu dosya sadece P1'den gelen bağlamı aktarır.
- P2 süresince `docs/p2/IMPLEMENTATION_NOTES.md` dosyası **çalışma boyunca** tutulacak (P1'deki gibi). Faz sonunda aynı arşivleme adımları tekrarlanır.
- GitHub Actions CI P2 kodunu da kapsayacak şekilde güncellenmesi gerekir (ör. PyTorch test job'u eklemek). Bu karar P2 Task'larında ele alınacak.
