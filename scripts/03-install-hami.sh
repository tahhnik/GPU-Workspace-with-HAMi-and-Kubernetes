#!/bin/bash
set -e

HAMI_DIR="${HAMI_DIR:-$HOME/HAMi}"
SPLIT_COUNT="${SPLIT_COUNT:-6}"

echo "=== Cloning HAMi repository ==="
if [ -d "$HAMI_DIR" ]; then
  echo "HAMi directory exists, pulling latest..."
  cd "$HAMI_DIR" && git pull
else
  git clone https://github.com/Project-HAMi/HAMi.git "$HAMI_DIR"
  cd "$HAMI_DIR"
fi

echo "=== Building Helm dependencies ==="
helm dependency build charts/hami

echo "=== Creating RuntimeClass ==="
kubectl apply -f - <<EOF
apiVersion: node.k8s.io/v1
kind: RuntimeClass
metadata:
  name: nvidia
handler: nvidia
EOF

echo "=== Labeling GPU node ==="
kubectl label nodes $(hostname) gpu=on --overwrite

echo "=== Installing HAMi ==="
helm upgrade --install hami charts/hami \
  --namespace kube-system \
  --set devicePlugin.deviceMemoryScaling=1 \
  --set devicePlugin.deviceSplitCount=$SPLIT_COUNT \
  --set devicePlugin.runtimeClassName=nvidia

echo "=== Waiting for HAMi pods ==="
sleep 10
kubectl get pods -n kube-system | grep hami

echo ""
echo "HAMi installed with $SPLIT_COUNT slices per GPU!"
echo "Verify with: kubectl describe node \$(hostname) | grep nvidia.com"
