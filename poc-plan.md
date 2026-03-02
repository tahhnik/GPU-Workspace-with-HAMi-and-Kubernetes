# PoC Plan: GPU Workspace with HAMi vGPU Slicing

## Objective

Validate that **HAMi (Heterogeneous AI Computing Middleware)** can provide:
- Software-enforced VRAM limits (8GB per student)
- Multi-tenant GPU sharing on RTX 4090
- Persistent storage isolation

---

## Success Criteria

| # | Criteria | How to Verify | Status |
|---|----------|---------------|--------|
| 1 | 3+ students run simultaneously on same GPU | Create 3 pods, all reach Running state | ✅ |
| 2 | Each sees only 8GB max VRAM | `nvidia-smi` shows 8GB limit inside pod | ✅ |
| 3 | CUDA OOM when exceeding limit | PyTorch allocation > 8GB fails | ✅ |
| 4 | Files persist after restart | Write file → delete pod → recreate → file exists | ⬜ |
| 5 | Max 1 GPU pod per namespace | ResourceQuota blocks 2nd pod | ⬜ |

---

## Architecture (PoC Minimal)

```
┌─────────────────────────────────────────────────────────────────┐
│                   GPU Node (2x RTX 4090)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   HAMi Components                        │   │
│  │                                                          │   │
│  │  hami-scheduler    → GPU-aware pod scheduling            │   │
│  │  hami-device-plugin → Exposes vGPU resources             │   │
│  │  HAMi-core (libvgpu.so) → CUDA memory enforcement        │   │
│  │                                                          │   │
│  │  Exposes: nvidia.com/gpu                                 │   │
│  │           nvidia.com/gpumem (VRAM in MiB)               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                         │                                       │
│     ┌───────────────────┼───────────────────┐                  │
│     ▼                   ▼                   ▼                  │
│  ┌───────┐          ┌───────┐          ┌───────┐              │
│  │Student│          │Student│          │Student│              │
│  │   A   │          │   B   │          │   C   │              │
│  │ 8GB   │          │ 8GB   │          │ 8GB   │              │
│  └───────┘          └───────┘          └───────┘              │
│                                                                 │
│  2x RTX 4090 (48GB each) → 12 concurrent 8GB sessions          │
│  (6 slices per GPU)                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Infrastructure Setup

### 1.1 Prerequisites on GPU Node

```bash
# Verify NVIDIA driver
nvidia-smi

# Expected output should show:
# - Driver Version: 565.57.01 (or similar)
# - CUDA Version: 12.7 (or similar)
# - 2x NVIDIA GeForce RTX 4090
```

### 1.2 Install K3s (Single Node for PoC)

```bash
# Install K3s
curl -sfL https://get.k3s.io | sh -

# Verify
sudo kubectl get nodes

# Set up kubeconfig for non-root access
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $(id -u):$(id -g) ~/.kube/config
export KUBECONFIG=~/.kube/config
```

### 1.3 Install NVIDIA GPU Operator

The GPU Operator handles NVIDIA Container Toolkit configuration automatically.

```bash
# Add NVIDIA Helm repo
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

# Install GPU Operator (with device plugin disabled - HAMi will handle that)
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace \
  --set driver.enabled=false \
  --set devicePlugin.enabled=false \
  --set toolkit.enabled=true

# Verify GPU Operator pods
kubectl get pods -n gpu-operator

# Wait for toolkit to be ready (this configures containerd for NVIDIA)
kubectl wait --for=condition=Ready pods -l app=nvidia-container-toolkit-daemonset \
  -n gpu-operator --timeout=300s
```

### 1.4 Create RuntimeClass for NVIDIA

```bash
cat <<EOF | kubectl apply -f -
apiVersion: node.k8s.io/v1
kind: RuntimeClass
metadata:
  name: nvidia
handler: nvidia
EOF

# Verify
kubectl get runtimeclass
```

### 1.5 Install HAMi

```bash
# Clone HAMi repo
git clone https://github.com/Project-HAMi/HAMi.git
cd HAMi

# Build helm dependencies
helm dependency build charts/hami

# Label GPU node (required for device plugin scheduling)
kubectl label nodes $(hostname) gpu=on

# Install HAMi via Helm
helm install hami charts/hami \
  --namespace kube-system \
  --set devicePlugin.deviceMemoryScaling=1 \
  --set devicePlugin.deviceSplitCount=6 \
  --set devicePlugin.runtimeClassName=nvidia

# Verify HAMi pods
kubectl get pods -n kube-system | grep hami

# Expected:
# hami-device-plugin-xxxxx   Running
# hami-scheduler-xxxxx       Running
```

### 1.6 Verify GPU Resources

```bash
# Check node has vGPU resources
kubectl describe node $(hostname) | grep -A 20 "Allocatable"

# Should show:
# nvidia.com/gpu: 12       (6 slices × 2 GPUs)
# nvidia.com/gpumem: ...   (memory resources)

# Check HAMi device plugin logs
kubectl logs -n kube-system -l app=hami-device-plugin --tail=50
```

---

## Phase 2: Student Namespace Setup

### 2.1 Create Namespace Template

Create file `templates/student-namespace.yaml`:

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: student-${USERNAME}
  labels:
    type: student-workspace
---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gpu-quota
  namespace: student-${USERNAME}
spec:
  hard:
    pods: "2"
    requests.cpu: "4"
    requests.memory: "16Gi"
    limits.cpu: "4"
    limits.memory: "16Gi"
    nvidia.com/gpu: "1"           # Max 1 GPU slice
    nvidia.com/gpumem: "8192"     # Max 8GB vGPU
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: home
  namespace: student-${USERNAME}
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 30Gi
  storageClassName: local-path
```

### 2.2 Create Test Students

```bash
# Create templates directory
mkdir -p templates

# Create 3 test student namespaces
for user in alice bob carol; do
  cat templates/student-namespace.yaml | sed "s/\${USERNAME}/$user/g" | kubectl apply -f -
done

# Verify
kubectl get ns | grep student
kubectl get pvc -A | grep student
kubectl get resourcequota -A | grep student
```

---

## Phase 3: GPU Workspace Pod

### 3.1 Workspace Pod Template

Create file `templates/workspace-pod.yaml`:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: workspace
  namespace: student-${USERNAME}
  labels:
    app: workspace
    type: gpu-session
spec:
  runtimeClassName: nvidia          # REQUIRED: Use NVIDIA container runtime
  schedulerName: hami-scheduler     # REQUIRED: Use HAMi scheduler
  containers:
  - name: workspace
    image: nvcr.io/nvidia/pytorch:24.01-py3
    command: ["sleep", "infinity"]
    resources:
      limits:
        nvidia.com/gpu: 1           # 1 GPU slice
        nvidia.com/gpumem: 8192     # 8GB VRAM limit (in MiB)
        memory: "8Gi"
        cpu: "2"
      requests:
        nvidia.com/gpu: 1
        nvidia.com/gpumem: 8192
        memory: "4Gi"
        cpu: "1"
    volumeMounts:
    - name: home
      mountPath: /home/student
    - name: dshm
      mountPath: /dev/shm
  volumes:
  - name: home
    persistentVolumeClaim:
      claimName: home
  - name: dshm
    emptyDir:
      medium: Memory
      sizeLimit: "2Gi"
  restartPolicy: Never
```

### 3.2 Deploy Test Workspaces

```bash
# Create workspace for alice
cat templates/workspace-pod.yaml | sed "s/\${USERNAME}/alice/g" | kubectl apply -f -

# Create workspace for bob
cat templates/workspace-pod.yaml | sed "s/\${USERNAME}/bob/g" | kubectl apply -f -

# Create workspace for carol
cat templates/workspace-pod.yaml | sed "s/\${USERNAME}/carol/g" | kubectl apply -f -

# Check status
kubectl get pods -A | grep workspace

# Expected output:
# student-alice   workspace   Running
# student-bob     workspace   Running
# student-carol   workspace   Running
```

---

## Phase 4: Verification Tests

### 4.1 Test: VRAM Limit Visibility

```bash
# Exec into alice's workspace
kubectl exec -it -n student-alice workspace -- bash

# Inside container:
nvidia-smi

# Expected output should show:
# +-----------------------------------------------------------------------------------------+
# | NVIDIA-SMI 565.57.01              Driver Version: 565.57.01      CUDA Version: 12.7     |
# |-----------------------------------------+------------------------+----------------------+
# | GPU  Name                 ...           |        Memory-Usage    |                      |
# |=========================================+========================+======================|
# |   0  NVIDIA GeForce RTX 4090            |    0MiB /  8192MiB     |  <-- 8GB limit shown |
# +-----------------------------------------+------------------------+----------------------+
```

### 4.2 Test: CUDA Memory Enforcement

```bash
# Inside container:
python3 << 'EOF'
import torch

# Check visible memory
print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Try to allocate within limit (should work)
x = torch.randn(1000, 1000, 1000, device='cuda')  # ~4GB
print("4GB allocation: SUCCESS")
del x
torch.cuda.empty_cache()

# Try to allocate over limit (should fail)
try:
    y = torch.randn(2000, 2000, 2000, device='cuda')  # ~32GB
    print("Over-limit allocation: UNEXPECTED SUCCESS")
except RuntimeError as e:
    print(f"Over-limit allocation: BLOCKED - {e}")
EOF
```

**Expected Result:**
- 4GB allocation succeeds
- 32GB allocation fails with CUDA OOM error
- HAMi-core enforces the limit at CUDA driver level

### 4.3 Test: Storage Persistence

```bash
# In alice's workspace
kubectl exec -it -n student-alice workspace -- bash -c "echo 'Hello from Alice' > /home/student/test.txt"

# Delete pod
kubectl delete pod -n student-alice workspace

# Recreate pod
cat templates/workspace-pod.yaml | sed "s/\${USERNAME}/alice/g" | kubectl apply -f -

# Wait for ready
kubectl wait --for=condition=Ready pod/workspace -n student-alice --timeout=60s

# Verify file exists
kubectl exec -it -n student-alice workspace -- cat /home/student/test.txt
# Should print: "Hello from Alice"
```

### 4.4 Test: Single GPU Pod Enforcement

```bash
# Try to create second GPU pod in alice's namespace
cat << 'EOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: workspace-2
  namespace: student-alice
spec:
  runtimeClassName: nvidia
  schedulerName: hami-scheduler
  containers:
  - name: workspace
    image: nvcr.io/nvidia/pytorch:24.01-py3
    command: ["sleep", "infinity"]
    resources:
      limits:
        nvidia.com/gpu: 1
        nvidia.com/gpumem: 8192
EOF

# Check status - should be blocked by ResourceQuota
kubectl get pods -n student-alice
kubectl describe pod workspace-2 -n student-alice
# Should show: "exceeded quota" error
```

### 4.5 Test: Concurrent Sessions

```bash
# Verify all 3+ students running simultaneously
kubectl get pods -A | grep workspace

# Expected:
# student-alice   workspace   Running
# student-bob     workspace   Running
# student-carol   workspace   Running

# Check GPU allocation
kubectl describe node $(hostname) | grep -A 10 "Allocated resources"

# Check HAMi scheduling
kubectl logs -n kube-system -l app=hami-scheduler --tail=20
```

---

## Phase 5: Cleanup

```bash
# Delete all test workspaces
kubectl delete pod workspace -n student-alice
kubectl delete pod workspace -n student-bob
kubectl delete pod workspace -n student-carol

# (Optional) Delete namespaces
kubectl delete ns student-alice student-bob student-carol
```

---

## Verification Checklist

| Test | Command | Expected Result | Status |
|------|---------|-----------------|--------|
| HAMi pods running | `kubectl get pods -n kube-system \| grep hami` | All Running | ✅ |
| 3+ pods running | `kubectl get pods -A \| grep workspace` | 3+ Running | ✅ |
| VRAM shows 8GB | `nvidia-smi` inside pod | 8192MiB limit | ✅ |
| 4GB alloc works | PyTorch test | SUCCESS | ✅ |
| 32GB alloc blocked | PyTorch test | CUDA OOM | ✅ |
| File persists | Delete/recreate pod | File exists | ⬜ |
| 2nd GPU pod blocked | Create workspace-2 | Quota exceeded | ⬜ |

---

## Troubleshooting

### HAMi device plugin not scheduling (DESIRED=0)

```bash
# Check if node has required label
kubectl get nodes --show-labels | grep gpu

# Add label if missing
kubectl label nodes $(hostname) gpu=on

# Restart device plugin
kubectl rollout restart daemonset hami-device-plugin -n kube-system
```

### Container runtime not configured for NVIDIA

If pods fail with `libcuda.so.1: cannot open shared object file`:

```bash
# Ensure GPU Operator toolkit is running
kubectl get pods -n gpu-operator | grep toolkit

# Ensure RuntimeClass exists
kubectl get runtimeclass nvidia

# Verify pod has runtimeClassName: nvidia in spec
kubectl get pod <pod-name> -n <namespace> -o yaml | grep runtimeClassName
```

### HAMi device plugin CrashLoopBackOff

```bash
# Check logs
kubectl logs -n kube-system -l app=hami-device-plugin

# Common issue: "Incompatible strategy detected"
# Fix: Install GPU Operator toolkit first, then HAMi

# Restart after fixing
kubectl rollout restart daemonset hami-device-plugin -n kube-system
```

### Pod stuck in Pending

```bash
kubectl describe pod <pod-name> -n <namespace>
# Check Events section for scheduling errors

# Common issues:
# - "0/1 nodes are available: insufficient nvidia.com/gpu"
#   → Check if HAMi device plugin is running and GPU resources are available
# - "no GPU available"
#   → All GPU slices are in use, wait or increase deviceSplitCount
```

### vGPU resources not visible on node

```bash
# Check HAMi device plugin pods
kubectl get pods -n kube-system | grep hami-device-plugin

# Check device plugin logs
kubectl logs -n kube-system -l app=hami-device-plugin --tail=100

# Verify GPU detection
kubectl exec -it -n kube-system $(kubectl get pods -n kube-system -l app=hami-device-plugin -o jsonpath='{.items[0].metadata.name}') -- nvidia-smi
```

---

## Key Configuration Notes

### HAMi Resource Names (NOT Volcano)
- `nvidia.com/gpu` - Number of GPU slices
- `nvidia.com/gpumem` - VRAM limit in MiB

### Required Pod Spec Fields
```yaml
spec:
  runtimeClassName: nvidia      # REQUIRED for GPU access
  schedulerName: hami-scheduler # REQUIRED for HAMi VRAM enforcement
```

### HAMi Helm Values
```bash
--set devicePlugin.deviceMemoryScaling=1    # No overcommit
--set devicePlugin.deviceSplitCount=6       # 6 slices per GPU
--set devicePlugin.runtimeClassName=nvidia  # Match RuntimeClass name
```

---

## Files Structure

```
HAMi-k8s-POC/
├── poc-plan.md              # This file
├── PRD.md                   # Product requirements
├── CONTEXT-DOC.md           # Technical context
├── system-design.md         # Full system architecture
├── templates/
│   ├── student-namespace.yaml
│   └── workspace-pod.yaml
└── scripts/
    ├── setup-cluster.sh
    ├── create-student.sh
    └── run-tests.sh
```

---

## PoC Results Summary

**Test Environment:**
- Host: shobdo.ddns.net:3334
- GPUs: 2x NVIDIA GeForce RTX 4090 (48GB each)
- K3s: v1.31.5+k3s1
- HAMi: Latest (helm chart)
- GPU Operator: For NVIDIA Container Toolkit

**Verified Results:**
1. ✅ 4 concurrent GPU pods running (alice, bob, carol, test-gpu)
2. ✅ Each pod sees 8192 MiB (8GB) VRAM limit in nvidia-smi
3. ✅ HAMi-core enforces CUDA OOM when allocation exceeds limit
4. ✅ HAMi scheduler distributes pods across GPU slices

**Capacity:**
- 2 GPUs × 6 slices = 12 concurrent 8GB sessions possible
- With 4GB slices: up to 24 concurrent sessions

---

## Next Steps After PoC

If PoC succeeds:
1. Add JupyterLab to workspace image
2. Add Ingress for web access
3. Build backend API for session management
4. Add authentication
5. Scale to multiple GPU nodes (add Node 2 with RTX 3090 Ti)
