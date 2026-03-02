# Troubleshooting Guide

## Common Issues

### 1. HAMi Device Plugin Not Scheduling (DESIRED=0)

**Symptom:**
```bash
$ kubectl get daemonset -n kube-system | grep hami
hami-device-plugin   0   0   0   0   0   ...
```

**Cause:** Node missing required label.

**Solution:**
```bash
kubectl label nodes $(hostname) gpu=on
kubectl rollout restart daemonset hami-device-plugin -n kube-system
```

---

### 2. Container Can't Find libcuda.so.1

**Symptom:**
```
error while loading shared libraries: libcuda.so.1: cannot open shared object file
```

**Cause:** Pod not using NVIDIA container runtime.

**Solution:**

1. Ensure RuntimeClass exists:
```bash
kubectl get runtimeclass nvidia
```

2. If missing, create it:
```bash
kubectl apply -f templates/runtimeclass.yaml
```

3. Ensure pod spec includes:
```yaml
spec:
  runtimeClassName: nvidia
```

---

### 3. HAMi Device Plugin CrashLoopBackOff

**Symptom:**
```bash
$ kubectl get pods -n kube-system | grep hami-device-plugin
hami-device-plugin-xxxxx   0/1   CrashLoopBackOff
```

**Cause:** NVIDIA Container Toolkit not configured.

**Solution:**

1. Install GPU Operator first:
```bash
./scripts/02-install-gpu-operator.sh
```

2. Wait for toolkit to be ready:
```bash
kubectl wait --for=condition=Ready pods \
  -l app=nvidia-container-toolkit-daemonset \
  -n gpu-operator --timeout=300s
```

3. Restart HAMi:
```bash
kubectl rollout restart daemonset hami-device-plugin -n kube-system
```

---

### 4. Pod Stuck in Pending

**Symptom:**
```bash
$ kubectl get pods -n student-alice
workspace   0/1   Pending   0   5m
```

**Check Events:**
```bash
kubectl describe pod workspace -n student-alice
```

**Common Causes:**

#### a) No GPU resources available
```
0/1 nodes are available: insufficient nvidia.com/gpu
```
- All GPU slices are in use
- Wait for other pods to finish or increase `deviceSplitCount`

#### b) HAMi scheduler not running
```
0/1 nodes are available: no available GPU node
```
```bash
kubectl get pods -n kube-system | grep hami-scheduler
# If not running, check logs:
kubectl logs -n kube-system -l app=hami-scheduler
```

#### c) Wrong scheduler name
Ensure pod spec has:
```yaml
spec:
  schedulerName: hami-scheduler
```

---

### 5. nvidia-smi Shows Full GPU Memory

**Symptom:** Inside pod, `nvidia-smi` shows 24GB instead of 8GB limit.

**Cause:** HAMi not properly intercepting CUDA calls.

**Check:**
```bash
# Verify HAMi environment variables in pod
kubectl exec -n student-alice workspace -- env | grep -i gpu

# Should see:
# GPU_MEMORY_LIMIT=8192
# LD_PRELOAD=/usr/local/vgpu/libvgpu.so
```

**Solution:**
1. Ensure using `hami-scheduler`:
```yaml
spec:
  schedulerName: hami-scheduler
```

2. Restart the device plugin:
```bash
kubectl rollout restart daemonset hami-device-plugin -n kube-system
```

---

### 6. GPU Resources Not Visible on Node

**Symptom:**
```bash
$ kubectl describe node $(hostname) | grep nvidia.com
# No output
```

**Solution:**

1. Check device plugin status:
```bash
kubectl get pods -n kube-system | grep hami-device-plugin
kubectl logs -n kube-system -l app=hami-device-plugin --tail=50
```

2. Verify NVIDIA driver:
```bash
nvidia-smi
```

3. Restart device plugin:
```bash
kubectl rollout restart daemonset hami-device-plugin -n kube-system
```

---

### 7. ResourceQuota Blocking Pod Creation

**Symptom:**
```
Error: pods "workspace-2" is forbidden: exceeded quota
```

**Cause:** This is expected behavior! ResourceQuota limits each namespace to 1 GPU pod.

**If you need multiple GPU pods per user:** Edit the quota in `templates/student-namespace.yaml`:
```yaml
spec:
  hard:
    nvidia.com/gpu: "2"        # Allow 2 GPU pods
    nvidia.com/gpumem: "16384" # 16GB total
```

---

### 8. PVC Not Creating

**Symptom:**
```bash
$ kubectl get pvc -n student-alice
NAME   STATUS    VOLUME   CAPACITY   ...
home   Pending                       ...
```

**Cause:** Storage class not available.

**Solution:**

1. Check available storage classes:
```bash
kubectl get storageclass
```

2. K3s includes `local-path` by default. If missing:
```bash
kubectl apply -f https://raw.githubusercontent.com/rancher/local-path-provisioner/master/deploy/local-path-storage.yaml
```

---

## Diagnostic Commands

```bash
# Overall cluster status
kubectl get nodes
kubectl get pods -A | grep -E "hami|nvidia|gpu"

# HAMi components
kubectl get pods -n kube-system | grep hami
kubectl logs -n kube-system -l app=hami-device-plugin --tail=100
kubectl logs -n kube-system -l app=hami-scheduler --tail=100

# GPU resources
kubectl describe node $(hostname) | grep -A 20 "Allocatable"

# Pod debugging
kubectl describe pod <pod-name> -n <namespace>
kubectl logs <pod-name> -n <namespace>

# GPU Operator status
kubectl get pods -n gpu-operator
kubectl get clusterpolicy
```

## Getting Help

1. Check HAMi GitHub issues: https://github.com/Project-HAMi/HAMi/issues
2. Check GPU Operator docs: https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/
3. K3s documentation: https://docs.k3s.io/
