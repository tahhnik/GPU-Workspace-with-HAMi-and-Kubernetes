#!/bin/bash
set -e

echo "=== Installing K3s ==="
curl -sfL https://get.k3s.io | sh -

echo "=== Setting up kubeconfig ==="
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $(id -u):$(id -g) ~/.kube/config
export KUBECONFIG=~/.kube/config

echo "=== Verifying K3s installation ==="
kubectl get nodes

echo ""
echo "K3s installed successfully!"
echo "Run: export KUBECONFIG=~/.kube/config"
