# Run locally with kind

## Prerequisites

Install Docker Desktop, [`kind`](https://kind.sigs.k8s.io/), and `kubectl`.
Ensure Docker Desktop is running.

## Deploy

Run these commands from the repository root in PowerShell:

```powershell
kind create cluster --name rag-agent
docker build -t rag-agent:local .
kind load docker-image rag-agent:local --name rag-agent
kubectl apply -f k8s/namespace.yaml
kubectl -n rag-agent create secret generic rag-agent-secrets --from-literal=GROQ_API_KEY='replace-with-your-key'
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
kubectl -n rag-agent rollout status deployment/rag-agent
kubectl -n rag-agent port-forward service/rag-agent 5000:5000
```

Open <http://localhost:5000> while the final command is running.

## Validate and troubleshoot

```powershell
kubectl -n rag-agent get pods,service
kubectl -n rag-agent logs deployment/rag-agent
curl http://localhost:5000/api/health
```

## Rebuild after code changes

```powershell
docker build -t rag-agent:local .
kind load docker-image rag-agent:local --name rag-agent
kubectl -n rag-agent rollout restart deployment/rag-agent
kubectl -n rag-agent rollout status deployment/rag-agent
```

`kind` is intended for local development. This setup stores uploads and generated indexes inside the Pod's writable filesystem, so they are lost when the Pod is recreated. Add a PersistentVolumeClaim backed by a suitable storage class before using it for durable data.

## Remove the local cluster

```powershell
kind delete cluster --name rag-agent
```
