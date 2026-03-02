#!/bin/bash
set -e

echo "=== Adding NVIDIA Helm repo ==="
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

echo "=== Installing GPU Operator ==="
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace \
  --set driver.enabled=false \
  --set devicePlugin.enabled=false \
  --set toolkit.enabled=true

echo "=== Waiting for GPU Operator pods ==="
kubectl wait --for=condition=Ready pods -l app.kubernetes.io/component=gpu-operator \
  -n gpu-operator --timeout=120s || true

echo "=== Waiting for NVIDIA Container Toolkit ==="
echo "This may take a few minutes..."
sleep 30

kubectl get pods -n gpu-operator

echo ""
echo "GPU Operator installed!"
echo "Wait for nvidia-container-toolkit pods to be Running before proceeding."
