# syntax=docker/dockerfile:1.6
# ============================================================================
# Stage 1 — builder: bağımlılıkları user-site'a kur, stage 2'ye kopyalanabilir
# ============================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .

# --user    : site-packages yerine ~/.local'a kurar → stage 2'ye taşınabilir
# --no-cache-dir : pip wheel cache'ini imaja yazma, boyut küçük kalsın
RUN pip install --user --no-cache-dir -r requirements.txt


# ============================================================================
# Stage 2 — production: minimal runtime + non-root user + healthcheck
# ============================================================================
FROM python:3.11-slim

# curl — HEALTHCHECK için gerekli. python:slim imajında default yok.
# --no-install-recommends + apt list temizliği → ek boyut minimize.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user — principle of least privilege. UID 1000 host user ile uyumlu
# (local dev'de volume mount edildiğinde permission sorunları engellenir).
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Builder'dan user-site paketlerini kopyala.
# NOT: SPEC "/root/.local"a kopyala diyor, ama stage 2'de USER appuser →
# appuser /root/'a erişemez (0700). Bunu /home/appuser/.local yaparak çözdük
# (bkz. docs/p1/IMPLEMENTATION_NOTES.md → Task 5).
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# Uygulama kodu — ownership appuser
COPY --chown=appuser:appuser app/ app/

# User-installed script'leri (uvicorn vs.) PATH'e ekle.
# PYTHONDONTWRITEBYTECODE : .pyc yazma — read-only FS + imaj boyutu
# PYTHONUNBUFFERED        : stdout/stderr buffer'lama — log akışı anlık
ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER appuser

EXPOSE 8000

# HEALTHCHECK — container sağlığı hakkında docker engine'e sinyal verir.
# --start-period=5s : startup grace period (uvicorn init süresi)
# --retries=3       : 3 ardışık başarısızlık → unhealthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production: 2 worker, no reload. Dev override'u docker-compose.yml'de.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
