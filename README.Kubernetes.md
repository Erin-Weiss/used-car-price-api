# Kubernetes Deployment

Production-ready Kubernetes manifests for the Used Car Price API, designed for reliable ML model serving with health monitoring, auto-scaling, and zero-downtime deployments.

---

## Architecture Overview

```
                         ┌──────────────────────────────────────────────┐
                         │           Kubernetes Cluster                 │
                         │     namespace: used-car-price-api            │
                         │                                              │
                         │   ┌──────────┐    ┌───────────────────────┐  │
  Incoming traffic ──────┼──▶│ Service  │───▶│  Deployment (2 pods)  │  │
                         │   │ :80      │    │                       │  │
                         │   └──────────┘    │  ┌─────┐   ┌─────┐    │  │
                         │                   │  │Pod 1│   │Pod 2│    │  │
                         │   ┌──────────┐    │  │:8000│   │:8000│    │  │
                         │   │   HPA    │───▶│  └─────┘   └─────┘    │  │
                         │   │ 2-6 pods │    │                       │  │
                         │   └──────────┘    └───────────────────────┘  │
                         │                                              │
                         │   ┌──────────┐                               │
                         │   │ConfigMap │  environment variables        │
                         │   └──────────┘                               │
                         └──────────────────────────────────────────────┘
```

The Service receives traffic and load-balances across healthy Pods. Each Pod runs the FastAPI application with the CatBoost model loaded in memory. The Horizontal Pod Autoscaler watches CPU utilization and scales between 2 and 6 replicas as demand changes.

---

## Manifest Reference

All manifests live in `k8s/` and are designed to be applied in sequence or all at once.

| File | Resource | Purpose |
|---|---|---|
| `namespace.yaml` | Namespace | Isolates all resources under `used-car-price-api` |
| `configmap.yaml` | ConfigMap | Non-secret environment configuration injected into Pods |
| `deployment.yaml` | Deployment | Pod spec with probes, resource limits, security, and update strategy |
| `service.yaml` | Service | Stable network endpoint with load balancing (NodePort for local dev) |
| `hpa.yaml` | HorizontalPodAutoscaler | CPU-based auto-scaling with scale-down stabilization |

---

## Key Design Decisions

### Health Probes

The API exposes two operational endpoints that map directly to Kubernetes probes:

| Endpoint | Probe Type | What It Checks |
|---|---|---|
| `GET /health` | Liveness | Process is running and responsive |
| `GET /ready` | Readiness + Startup | CatBoost model and all JSON artifacts are loaded in memory |

**Why the startup probe hits `/ready` instead of `/health`:** The CatBoost model takes time to load on startup. `/health` returns `200` as soon as the FastAPI process starts — before the model is in memory. `/ready` returns `200` only after `load_runtime_state()` completes. The startup probe allows up to ~125 seconds for this initialization (`5s initial delay + 24 failures × 5s period`), preventing Kubernetes from killing a slow-starting Pod while still catching actual startup crashes.

### Rolling Updates

```yaml
strategy:
  type: RollingUpdate
  rollingUpdate:
    maxUnavailable: 0
    maxSurge: 1
```

`maxUnavailable: 0` ensures no running Pod is terminated until its replacement passes all health checks. Combined with 2 replicas, this guarantees at least 2 Pods serve traffic at all times during a deployment. `maxSurge: 1` limits the rollout to creating one extra Pod at a time, keeping resource consumption predictable.

### Resource Sizing

```yaml
resources:
  requests:
    cpu: "250m"
    memory: "512Mi"
  limits:
    cpu: "1000m"
    memory: "1Gi"
```

The CatBoost model and JSON artifacts live entirely in memory once loaded. Requests are set to accommodate the loaded model at rest; limits provide headroom for inference-time CPU bursts and pandas DataFrame construction during feature engineering. These values are sized for local development with Minikube — production deployments should profile actual usage and adjust upward (512m–1000m CPU requests, 1–2Gi memory requests).

### Container Security

```yaml
securityContext:
  allowPrivilegeEscalation: false
  runAsNonRoot: true
  runAsUser: 10001
  capabilities:
    drop:
      - ALL
```

The container runs as a non-root user (`appuser`, UID 10001, defined in the Dockerfile), drops all Linux capabilities, and prevents privilege escalation. This limits the blast radius if the container is compromised — the process cannot modify the host filesystem, bind to privileged ports, or escalate to root.

### Auto-Scaling

The HPA scales from 2 to 6 replicas based on average CPU utilization across all Pods, with a 70% target threshold. Scale-down behavior includes a 120-second stabilization window and a limit of removing one Pod per 60 seconds to prevent thrashing after traffic spikes.

---

## Local Deployment (Minikube)

### Prerequisites

- [Docker Desktop](https://docs.docker.com/get-docker/) — container runtime
- [kubectl](https://kubernetes.io/docs/tasks/tools/) — Kubernetes CLI
- [Minikube](https://minikube.sigs.k8s.io/docs/start/) — local Kubernetes cluster

### Deploy

```bash
# Ensure Docker Desktop is running (Minikube uses it as its driver).
# On macOS, if it's not already open:
open -a Docker

# Start a local cluster with enough resources for the ML model
minikube start --driver=docker --memory=4096 --cpus=2

# Build the Docker image inside Minikube's Docker environment
eval $(minikube docker-env)
docker build -t used-car-price-api:latest .

# Apply all manifests
kubectl apply -f k8s/

# Wait for Pods to become ready (model loading takes ~30-60s)
kubectl wait --for=condition=ready pod \
  -l app.kubernetes.io/name=used-car-price-api \
  -n used-car-price-api \
  --timeout=120s

# Get the URL
minikube service used-car-price-api -n used-car-price-api --url
```

### Verify

```bash
# Health check
curl http://<minikube-url>/health

# Readiness check
curl http://<minikube-url>/ready

# Prediction
curl -X POST http://<minikube-url>/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "manufacturer": "toyota",
    "model": "camry le",
    "year": 2020,
    "mileage": 35000,
    "engine": "2.5l i4 dohc 16v",
    "transmission": "8 speed automatic",
    "drivetrain": "fwd",
    "fuel_type": "gasoline",
    "exterior_color": "silver metallic",
    "interior_color": "black leather",
    "accidents_or_damage": 0,
    "one_owner": 1,
    "personal_use_only": 1
  }'

# Check HPA status
kubectl get hpa -n used-car-price-api

# View Pod logs
kubectl logs -l app.kubernetes.io/name=used-car-price-api \
  -n used-car-price-api -f
```

### Tear Down
 
```bash
kubectl delete -f k8s/
 
# Pause the cluster (preserves state, restart with minikube start)
minikube stop
 
# Or remove the cluster entirely to free disk space
minikube delete
```

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `ImagePullBackOff` | Image not built inside Minikube's Docker | Run `eval $(minikube docker-env)` then rebuild |
| `CrashLoopBackOff` | App crashing on startup | Check logs: `kubectl logs <pod-name> -n used-car-price-api --previous` |
| Pods `0/1 Running` | Model still loading | Wait for startup probe to pass (~30-60s), or check `/ready` response |
| HPA shows `<unknown>/70%` | Metrics server not enabled | Run `minikube addons enable metrics-server` |

---

## Production Considerations

The current manifests target local development. For a production deployment, consider:

- **Service type**: Switch from `NodePort` to `ClusterIP` behind an Ingress controller with TLS termination.
- **Image registry**: Replace `used-car-price-api:latest` with a versioned tag from a container registry (e.g. `ghcr.io/<user>/used-car-price-api:0.1.0`).
- **Resource limits**: Profile actual memory and CPU usage under load and adjust requests/limits accordingly.
- **Secrets**: If credentials are added in the future, use Kubernetes Secrets (or an external secrets manager) instead of the ConfigMap.
- **Monitoring**: Add Prometheus annotations and a `/metrics` endpoint for observability.