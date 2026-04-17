# P4 — Production'a Çıkış: Konsept + Spesifikasyon + Görevler

---

## Konsept: neden cloud + Kubernetes?

P1-P3'te her şey `localhost`'ta çalıştı. Gerçek dünyada:
- Başka biri de kullanmak istiyor → internet erişimi lazım
- Aynı anda 100 öğrenci soru soruyor → tek instance yetmez
- Sunucu çöktü → otomatik restart lazım
- Yeni versiyon çıktı → sıfır kesinti ile güncelleme lazım

Bunların hepsi Kubernetes'in çözdüğü problemler.

### Kavramlar

**Kubernetes (K8s):** Container orchestration sistemi. "Bu container'ı 3 kopya çalıştır, biri çökerse yerine yenisini başlat, trafiki aralarında dağıt" komutlarını anlayan sistem.

**Pod:** K8s'in en küçük birimi. İçinde 1+ container barınır.

**Deployment:** "Bu pod'dan kaç tane olsun, nasıl güncellenmeli" tarifi.

**Service:** Pod'lara dış dünyadan nasıl erişileceği. Load balancer gibi davranır.

**Ingress:** HTTP routing. "api.eduai.com/v1/ gelince API pod'una yönlendir"

**Helm:** K8s için paket manager. Tüm manifest'leri tek komutla deploy eder.

**AWS EKS:** Amazon'un yönetilen Kubernetes servisi. Cluster kurulumu AWS yapıyor, sen sadece uygulama deploy ediyorsun.

**Terraform:** "Infrastructure as Code." AWS kaynakları (EKS, VPC, ECR) YAML/HCL dosyalarıyla tanımlanıyor, `terraform apply` komutuyla gerçeğe dönüşüyor.

**Prometheus + Grafana:** Metrik toplama ve görselleştirme. "API kaç saniyede cevap veriyor? Hata oranı nedir?" soruları buradan cevaplanıyor.

---

## Klasör yapısı

```
eduai-platform/
└── infra/
    ├── terraform/
    │   ├── main.tf              ← AWS provider, EKS cluster
    │   ├── variables.tf         ← parametreler
    │   ├── outputs.tf           ← kubeconfig, endpoint
    │   └── ecr.tf               ← Docker image registry
    ├── k8s/
    │   ├── base/
    │   │   ├── api-deployment.yaml
    │   │   ├── api-service.yaml
    │   │   ├── qdrant-deployment.yaml
    │   │   ├── qdrant-service.yaml
    │   │   └── ingress.yaml
    │   └── overlays/
    │       ├── dev/
    │       │   └── kustomization.yaml   ← 1 replica, debug mode
    │       └── prod/
    │           └── kustomization.yaml   ← 3 replicas, resource limits
    ├── helm/
    │   └── eduai/
    │       ├── Chart.yaml
    │       ├── values.yaml
    │       └── templates/               ← base/ altındaki manifest'ler şablonlanmış
    └── monitoring/
        ├── prometheus-values.yaml
        └── grafana/
            └── dashboards/
                └── eduai-overview.json
```

---

## Görev Listesi

### Task 0 — AWS hesabı + araçlar ⏱ ~30 dk

**Gereksinimler:**
- AWS Free Tier hesabı (eduai.vizyon.edu.tr için yeni hesap açılabilir)
- AWS CLI kurulumu + konfigürasyonu
- kubectl kurulumu
- Terraform kurulumu
- Helm kurulumu

### Claude Code'a ver:
```
Proje: eduai-platform/infra/

Görev: Terraform ile AWS EKS cluster oluştur.

terraform/main.tf:
- AWS provider (region: eu-central-1 — Frankfurt, Türkiye'ye yakın)
- VPC: 2 public, 2 private subnet
- EKS cluster: version 1.29
  - node group: t3.medium, min=1, max=3, desired=2
  - node group adı: "eduai-nodes"
- ECR repository: "eduai-api" (Docker image'ları buraya push edilecek)

terraform/variables.tf:
- cluster_name = "eduai-platform"
- aws_region = "eu-central-1"
- environment = "dev"

Tüm kaynaklar "eduai" prefix ile tag'lensin.
```

### Bunu kendin yap:
```bash
cd infra/terraform
terraform init
terraform plan    # önce ne yapacağını gör
terraform apply   # onay ver

# Kubeconfig al:
aws eks update-kubeconfig --region eu-central-1 --name eduai-platform
kubectl get nodes  # node'lar hazır mı?

git add . && git commit -m "infra: Terraform EKS cluster setup"
```

---

### Task 1 — Docker image'ları ECR'a push ⏱ ~30 dk

### Claude Code'a ver:
```
Proje: eduai-platform/

Görev: GitHub Actions workflow'unu güncelle — CI'a CD ekle.

.github/workflows/deploy.yml (yeni dosya):
Tetikleyici: main branch'e push

Jobs:
1. build-and-push:
   - Docker image build: services/api/
   - Tag: {ECR_URL}/eduai-api:{github.sha}
   - ECR'a push (aws-actions/amazon-ecr-login kullan)

Secrets (GitHub'da tanımlanacak):
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY  
- ECR_REGISTRY (terraform output'tan alınır)
```

---

### Task 2 — Kubernetes manifest'leri ⏱ ~45 dk

### Claude Code'a ver:
```
Proje: eduai-platform/infra/k8s/

Görev: Tüm K8s manifest'lerini yaz.

base/api-deployment.yaml:
- replicas: 2
- image: {ECR_URL}/eduai-api:latest
- resources:
    requests: cpu=100m, memory=256Mi
    limits: cpu=500m, memory=512Mi
- livenessProbe: GET /health, 30s delay
- readinessProbe: GET /health, 10s delay
- env: DEBUG=false, LOG_LEVEL=info

base/api-service.yaml:
- type: ClusterIP
- port: 80 → targetPort: 8000

base/qdrant-deployment.yaml:
- replicas: 1
- image: qdrant/qdrant:latest
- PersistentVolumeClaim: 10Gi

base/ingress.yaml:
- host: api.eduai-platform.com (placeholder)
- /v1/ → api-service:80

overlays/dev/kustomization.yaml:
- replicas: 1
- DEBUG=true patch

overlays/prod/kustomization.yaml:
- replicas: 3
- resource limits artır
```

### Bunu kendin yap:
```bash
# Dev ortamına deploy:
kubectl apply -k infra/k8s/overlays/dev/

# Kontrol:
kubectl get pods
kubectl get services
kubectl logs deployment/eduai-api

# API'ye eriş (port-forward ile test):
kubectl port-forward service/api-service 8080:80
curl http://localhost:8080/health

git add . && git commit -m "infra: Kubernetes manifests for EKS deployment"
```

---

### Task 3 — Helm chart ⏱ ~30 dk

### Claude Code'a ver:
```
Proje: eduai-platform/infra/helm/eduai/

Görev: K8s manifest'lerini Helm chart'a dönüştür.

Chart.yaml:
- name: eduai
- version: 0.1.0
- appVersion: "1.0"

values.yaml:
- api.image.repository: {ECR_URL}/eduai-api
- api.image.tag: latest
- api.replicaCount: 2
- api.resources: (requests/limits)
- qdrant.persistence.size: 10Gi
- ingress.enabled: true
- ingress.host: api.eduai-platform.com

templates/: base/ manifest'lerini {{ .Values.xxx }} ile şablonla

Helm ile deploy test et:
helm install eduai ./infra/helm/eduai --namespace eduai --create-namespace
```

---

### Task 4 — Monitoring ⏱ ~45 dk

### Claude Code'a ver:
```
Proje: eduai-platform/infra/monitoring/

Görev: Prometheus + Grafana kur ve EduAI dashboard'u oluştur.

1. Prometheus Helm ile kur:
   helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
   helm install prometheus prometheus-community/kube-prometheus-stack

2. FastAPI'ye Prometheus metrikleri ekle (services/api/app/main.py):
   - prometheus-fastapi-instrumentator kütüphanesi
   - /metrics endpoint'i expose et
   - Custom metrikler:
     * questions_total (counter): toplam soru sayısı
     * question_latency_seconds (histogram): cevap süresi
     * rag_retrieval_latency_seconds (histogram): RAG süresi

3. Grafana dashboard JSON (grafana/dashboards/eduai-overview.json):
   - Panel 1: Toplam soru sayısı (last 24h)
   - Panel 2: API latency percentiles (p50, p95, p99)
   - Panel 3: RAG retrieval süresi
   - Panel 4: HTTP error rate
   - Panel 5: Pod CPU/memory kullanımı
```

### Bunu kendin yap:
```bash
# Grafana'ya eriş:
kubectl port-forward service/prometheus-grafana 3000:80
# http://localhost:3000 (admin/prom-operator)
# Dashboard'u import et

# Birkaç test isteği at:
for i in {1..20}; do curl http://localhost:8080/health; done

# Metriklerin Grafana'da göründüğünü doğrula
git add . && git commit -m "feat: Prometheus + Grafana monitoring"
```

---

### Task 5 — AWS Solutions Architect cert track ⏱ Paralel

Bu P4 ile paralel giden cert hazırlığı. Pratik AWS deneyimi zaten edindik — teoriyi tamamla.

**Ücretsiz kaynaklar:**
- AWS Skill Builder: "AWS Cloud Practitioner Essentials" (8 saat)
- Andrew Brown'un YouTube kursu: "AWS Solutions Architect Associate" (~30 saat)
- Whizlabs practice exams (ücretli ama değer)

**Sınav:** AWS SAA-C03, ~$150, Pearson VUE test center (İstanbul'da mevcut).

**Takvim önerisi:** P4 Tasks 0-3 bittikten sonra 4 hafta yoğun çalışma → sınav.

---

## P4 tamamlandı mı? Kontrol listesi

```
[ ] terraform apply → EKS cluster ayakta, kubectl get nodes çalışıyor
[ ] Docker image ECR'da, GitHub Actions otomatik push ediyor
[ ] kubectl get pods → tüm pod'lar Running
[ ] API internet'ten erişilebilir (Ingress veya port-forward)
[ ] kubectl scale deployment eduai-api --replicas=3 → sorunsuz
[ ] Grafana dashboard → metrikler görünüyor
[ ] Helm chart: helm upgrade ile yeni versiyon deploy ediliyor
[ ] README: "nasıl deploy edilir" tam kılavuz (terraform → helm → grafana)
```

---

## Son: GitHub'da ne görünecek?

```
github.com/yasirbeydili/eduai-platform

⭐ EduAI Platform — Turkish High School AI Q&A System

Stack: FastAPI · PyTorch · LangChain · LangGraph · CrewAI · 
       Qdrant · Docker · Kubernetes · AWS EKS · MLflow · Terraform

├── 13 haftalık production deployment
├── Fine-tuned LLM (Türkçe eğitim domain'i)
├── Multi-agent RAG pipeline
├── Kubernetes + AWS altyapısı
└── MLflow ile experiment tracking
```

Bu tek repo, bir recruiter'ın görmek istediği her şeyi içeriyor.
