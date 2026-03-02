#!/bin/bash
set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <username>"
  echo "Example: $0 alice"
  exit 1
fi

USERNAME="$1"
NAMESPACE="student-$USERNAME"

echo "=== Testing VRAM limit for $USERNAME ==="
echo ""

echo "1. Checking nvidia-smi output:"
kubectl exec -n "$NAMESPACE" workspace -- nvidia-smi

echo ""
echo "2. Running PyTorch VRAM test:"
kubectl exec -n "$NAMESPACE" workspace -- python3 -c "
import torch

print(f'CUDA Device: {torch.cuda.get_device_name(0)}')
print(f'Total VRAM visible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
print()

# Test allocation within limit
print('Testing 4GB allocation (within limit)...')
try:
    x = torch.randn(1000, 1000, 1000, device='cuda')
    print('  Result: SUCCESS')
    del x
    torch.cuda.empty_cache()
except RuntimeError as e:
    print(f'  Result: FAILED - {e}')

print()

# Test allocation over limit
print('Testing 32GB allocation (over limit)...')
try:
    y = torch.randn(2000, 2000, 2000, device='cuda')
    print('  Result: UNEXPECTED SUCCESS (limit not enforced!)')
    del y
except RuntimeError as e:
    print('  Result: BLOCKED by HAMi (expected)')
"

echo ""
echo "=== VRAM limit test complete ==="
