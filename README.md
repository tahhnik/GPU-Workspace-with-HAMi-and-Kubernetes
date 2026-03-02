# HAMi GPU Workspace

A multi-tenant GPU sharing platform using HAMi (Heterogeneous AI Computing Middleware) on Kubernetes. This project enables multiple users to share consumer-grade GPUs (like RTX 4090) with isolated VRAM allocations, making it possible to provide GPU workspaces to 50+ users on a small GPU cluster.

## Table of Contents

- [Problem Statement](#problem-statement)
- [Solution Overview](#solution-overview)
- [How HAMi Works](#how-hami-works)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [VRAM Enforcement Deep Dive](#vram-enforcement-deep-dive)
- [PoC Results](#poc-results)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)
- [References](#references)

---

## Problem Statement

### The Challenge

We want to provide GPU-accelerated workspaces (like Google Colab or Kaggle Notebooks) for students/users on our own infrastructure. The requirements are:

1. **Multiple users share the same GPU** - A single RTX 4090 (24GB VRAM) should support 3-6 concurrent users
2. **VRAM isolation** - Each user gets a fixed VRAM allocation (e.g., 8GB) that they cannot exceed
3. **File system isolation** - Each user has their own persistent storage
4. **Resource quotas** - Prevent any single user from consuming all resources

### Why Not Use NVIDIA MIG?

NVIDIA's Multi-Instance GPU (MIG) technology provides hardware-level GPU partitioning, but it has significant limitations:

| Feature | MIG | HAMi (This Solution) |
|---------|-----|----------------------|
| Supported GPUs | A100, A30, H100 (Data center only) | **All NVIDIA GPUs** including consumer cards |
| Hardware partitioning | Yes | No (software-based) |
| Minimum slice | 1/7 of GPU | **Any size** (1MB granularity) |
| Cost | $10,000+ per GPU | Works with $1,500 RTX 4090 |
| Flexibility | Fixed partitions | Dynamic allocation |

**Consumer GPUs (RTX 3090, 4090, etc.) do not support MIG.** This project uses HAMi to achieve similar multi-tenancy on consumer hardware.

### Why Not Use NVIDIA Time-Slicing?

NVIDIA's time-slicing allows multiple containers to share a GPU, but:

- **No VRAM isolation** - All containers see full GPU memory
- **No memory enforcement** - One container can OOM others
- **No fair scheduling** - Workloads compete for GPU time

HAMi provides actual VRAM limits enforced at the CUDA driver level.

---

## Solution Overview

This project uses a stack of technologies to provide isolated GPU workspaces:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          USER WORKSPACES                                │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │
│   │  Alice  │  │   Bob   │  │  Carol  │  │  David  │  │   Eve   │      │
│   │  8GB    │  │  8GB    │  │  8GB    │  │  4GB    │  │  4GB    │      │
│   └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘      │
│        │            │            │            │            │            │
├────────┴────────────┴────────────┴────────────┴────────────┴────────────┤
│                                                                         │
│   HAMi Layer (VRAM Enforcement)                                        │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │  libvgpu.so - Intercepts CUDA calls, enforces memory limits     │  │
│   │  hami-scheduler - Tracks VRAM allocations, schedules pods       │  │
│   │  hami-device-plugin - Exposes virtual GPU resources             │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Kubernetes Layer (Orchestration)                                     │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │  K3s - Lightweight Kubernetes distribution                       │  │
│   │  Namespaces - User isolation                                     │  │
│   │  ResourceQuotas - Resource limits per user                       │  │
│   │  PVCs - Persistent storage per user                              │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   NVIDIA Layer (GPU Access)                                            │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │  GPU Operator - Manages NVIDIA components in Kubernetes         │  │
│   │  Container Toolkit - Enables GPU access in containers           │  │
│   │  NVIDIA Driver - Hardware interface                             │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                         PHYSICAL HARDWARE                               │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │  RTX 4090 (48GB)  │  RTX 4090 (48GB)  │  RTX 3090 Ti (24GB)     │  │
│   └─────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Purpose |
|-----------|---------|
| **K3s** | Lightweight Kubernetes distribution (single binary, low overhead) |
| **GPU Operator** | NVIDIA's operator that manages container toolkit and drivers |
| **HAMi** | Heterogeneous AI computing Middleware - provides GPU virtualization |
| **HAMi Device Plugin** | Kubernetes device plugin that exposes `nvidia.com/gpu` and `nvidia.com/gpumem` resources |
| **HAMi Scheduler** | Custom Kubernetes scheduler that tracks VRAM allocations |
| **HAMi-core (libvgpu.so)** | Library injected into containers to intercept CUDA calls |

---

## How HAMi Works

### The Magic: CUDA Interception

HAMi enforces VRAM limits by intercepting CUDA API calls. Here's how it works:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    CUDA CALL FLOW (with HAMi)                           │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   User Application (PyTorch, TensorFlow, etc.)                          │
│        │                                                                 │
│        │ torch.randn(1000,1000,1000).cuda()                             │
│        ▼                                                                 │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                    libvgpu.so (HAMi-core)                        │   │
│   │                                                                  │   │
│   │  1. Intercepts cudaMalloc() call                                │   │
│   │  2. Checks: current_usage + requested <= GPU_MEMORY_LIMIT?      │   │
│   │     - If YES: Forward call to real CUDA driver                  │   │
│   │     - If NO: Return CUDA_ERROR_OUT_OF_MEMORY                    │   │
│   │  3. Tracks memory allocations                                   │   │
│   │  4. Reports fake total memory to cudaGetDeviceProperties()      │   │
│   │                                                                  │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│        │                                                                 │
│        ▼                                                                 │
│   Real NVIDIA CUDA Driver                                               │
│        │                                                                 │
│        ▼                                                                 │
│   Physical GPU Hardware                                                 │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Environment Variables Injected by HAMi

When a pod is scheduled by HAMi, these environment variables are set:

```bash
GPU_MEMORY_LIMIT=8192          # VRAM limit in MiB
CUDA_DEVICE_MEMORY_LIMIT=8192  # Same, for compatibility
LD_PRELOAD=/usr/local/vgpu/libvgpu.so  # Intercept library
```

The `LD_PRELOAD` mechanism causes `libvgpu.so` to be loaded before the real CUDA libraries, allowing it to intercept all CUDA calls.

### HAMi Scheduling Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        POD SCHEDULING FLOW                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. User creates Pod with:                                             │
│     resources:                                                         │
│       limits:                                                          │
│         nvidia.com/gpu: 1                                              │
│         nvidia.com/gpumem: 8192                                        │
│                                                                         │
│  2. Kubernetes API Server receives Pod                                 │
│        │                                                                │
│        ▼                                                                │
│  3. HAMi Scheduler (schedulerName: hami-scheduler)                     │
│     ┌───────────────────────────────────────────────────────────────┐  │
│     │ a. Query all GPUs on all nodes                                │  │
│     │ b. Check allocated VRAM per GPU (tracked in annotations)     │  │
│     │ c. Find GPU with: total_vram - allocated_vram >= 8192        │  │
│     │ d. Bind Pod to selected node/GPU                             │  │
│     │ e. Update GPU allocation tracking                            │  │
│     └───────────────────────────────────────────────────────────────┘  │
│        │                                                                │
│        ▼                                                                │
│  4. Kubelet on Node                                                    │
│     ┌───────────────────────────────────────────────────────────────┐  │
│     │ a. HAMi Device Plugin allocates GPU slice                    │  │
│     │ b. Sets environment variables (GPU_MEMORY_LIMIT, LD_PRELOAD) │  │
│     │ c. Mounts libvgpu.so into container                          │  │
│     │ d. Starts container with NVIDIA runtime                      │  │
│     └───────────────────────────────────────────────────────────────┘  │
│        │                                                                │
│        ▼                                                                │
│  5. Container starts with VRAM-limited GPU access                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### HAMi Resource Types

HAMi exposes these Kubernetes resources:

| Resource | Description | Example |
|----------|-------------|---------|
| `nvidia.com/gpu` | Number of GPU "slices" to allocate | `1` |
| `nvidia.com/gpumem` | VRAM limit in MiB | `8192` (8GB) |
| `nvidia.com/gpucores` | GPU compute percentage (optional) | `50` (50%) |

---

## Architecture

### Single Node Architecture (PoC)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      GPU SERVER (shobdo.ddns.net)                       │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                        K3s Control Plane                          │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐   │ │
│  │  │ API Server  │  │    etcd     │  │  Controller Manager     │   │ │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────┘   │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                      System Components                            │ │
│  │                                                                   │ │
│  │  ┌──────────────────┐  ┌──────────────────┐                      │ │
│  │  │  hami-scheduler  │  │ hami-device-     │                      │ │
│  │  │  (Deployment)    │  │ plugin (DS)      │                      │ │
│  │  │                  │  │                  │                      │ │
│  │  │ Tracks VRAM      │  │ Exposes GPU      │                      │ │
│  │  │ allocations,     │  │ resources,       │                      │ │
│  │  │ schedules pods   │  │ injects HAMi     │                      │ │
│  │  └──────────────────┘  └──────────────────┘                      │ │
│  │                                                                   │ │
│  │  ┌──────────────────────────────────────────────────────────┐    │ │
│  │  │              GPU Operator Components                      │    │ │
│  │  │  nvidia-container-toolkit-daemonset                       │    │ │
│  │  │  nvidia-operator-validator                                │    │ │
│  │  └──────────────────────────────────────────────────────────┘    │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                      User Namespaces                              │ │
│  │                                                                   │ │
│  │   student-alice          student-bob           student-carol      │ │
│  │  ┌─────────────┐       ┌─────────────┐       ┌─────────────┐     │ │
│  │  │ Pod:        │       │ Pod:        │       │ Pod:        │     │ │
│  │  │ workspace   │       │ workspace   │       │ workspace   │     │ │
│  │  │ 8GB VRAM    │       │ 8GB VRAM    │       │ 8GB VRAM    │     │ │
│  │  │             │       │             │       │             │     │ │
│  │  │ PyTorch     │       │ PyTorch     │       │ PyTorch     │     │ │
│  │  │ JupyterLab  │       │ JupyterLab  │       │ JupyterLab  │     │ │
│  │  ├─────────────┤       ├─────────────┤       ├─────────────┤     │ │
│  │  │ PVC: home   │       │ PVC: home   │       │ PVC: home   │     │ │
│  │  │ 30GB        │       │ 30GB        │       │ 30GB        │     │ │
│  │  └─────────────┘       └─────────────┘       └─────────────┘     │ │
│  │                                                                   │ │
│  │  ResourceQuota per namespace:                                    │ │
│  │  - nvidia.com/gpu: 1                                             │ │
│  │  - nvidia.com/gpumem: 8192                                       │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                      Physical Hardware                            │ │
│  │                                                                   │ │
│  │  GPU 0: RTX 4090              GPU 1: RTX 4090                    │ │
│  │  ┌───────────────────────┐   ┌───────────────────────────┐       │ │
│  │  │     48GB VRAM         │   │     48GB VRAM             │       │ │
│  │  │                       │   │                           │       │ │
│  │  │  ┌─────┐ ┌─────┐     │   │  ┌─────┐ ┌─────┐         │       │ │
│  │  │  │8GB  │ │8GB  │     │   │  │8GB  │ │8GB  │         │       │ │
│  │  │  │alice│ │bob  │     │   │  │carol│ │free │         │       │ │
│  │  │  └─────┘ └─────┘     │   │  └─────┘ └─────┘         │       │ │
│  │  │  ┌─────┐ ┌─────┐     │   │  ┌─────┐ ┌─────┐         │       │ │
│  │  │  │8GB  │ │8GB  │     │   │  │8GB  │ │8GB  │         │       │ │
│  │  │  │free │ │free │     │   │  │free │ │free │         │       │ │
│  │  │  └─────┘ └─────┘     │   │  └─────┘ └─────┘         │       │ │
│  │  │  ┌─────┐ ┌─────┐     │   │  ┌─────┐ ┌─────┐         │       │ │
│  │  │  │8GB  │ │8GB  │     │   │  │8GB  │ │8GB  │         │       │ │
│  │  │  │free │ │free │     │   │  │free │ │free │         │       │ │
│  │  │  └─────┘ └─────┘     │   │  └─────┘ └─────┘         │       │ │
│  │  └───────────────────────┘   └───────────────────────────┘       │ │
│  │                                                                   │ │
│  │  Total: 96GB VRAM = 12 × 8GB slices                              │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### User Isolation Model

Each user gets their own Kubernetes namespace with:

```yaml
# Namespace - isolation boundary
apiVersion: v1
kind: Namespace
metadata:
  name: student-alice

---
# ResourceQuota - prevents resource abuse
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gpu-quota
  namespace: student-alice
spec:
  hard:
    pods: "2"                    # Max 2 pods
    nvidia.com/gpu: "1"          # Max 1 GPU slice
    nvidia.com/gpumem: "8192"    # Max 8GB VRAM

---
# PersistentVolumeClaim - user's data persists across restarts
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: home
  namespace: student-alice
spec:
  accessModes: [ReadWriteOnce]
  resources:
    requests:
      storage: 30Gi
```

---

## Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 16GB | 32GB+ |
| Storage | 100GB SSD | 500GB+ NVMe |
| GPU | Any NVIDIA GPU with 8GB+ VRAM | RTX 3090/4090 |

### Software Requirements

| Software | Version | Notes |
|----------|---------|-------|
| OS | Ubuntu 20.04+ | Other Linux distros may work |
| NVIDIA Driver | 535+ | Required for CUDA 12.x support |
| Helm | 3.x | For installing GPU Operator and HAMi |
| curl, git | Any | For downloading components |

### Verify GPU Setup

Before starting, ensure your GPU is properly detected:

```bash
# Check NVIDIA driver
nvidia-smi

# Expected output:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 565.57.01    Driver Version: 565.57.01    CUDA Version: 12.7     |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |===============================+======================+======================|
# |   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0 Off |                  Off |
# | 30%   35C    P8    25W / 450W |      0MiB / 24564MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+
```

---

## Installation

### Step 1: Install K3s

K3s is a lightweight Kubernetes distribution perfect for GPU workloads.

```bash
# Run the installation script
./scripts/01-install-k3s.sh

# Or manually:
curl -sfL https://get.k3s.io | sh -

# Set up kubeconfig for non-root access
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $(id -u):$(id -g) ~/.kube/config
export KUBECONFIG=~/.kube/config

# Verify installation
kubectl get nodes
# NAME                STATUS   ROLES                  AGE   VERSION
# your-hostname       Ready    control-plane,master   1m    v1.31.5+k3s1
```

**What this does:**
- Installs K3s as a systemd service
- Configures containerd as the container runtime
- Sets up local storage provisioner (local-path)
- Starts the Kubernetes API server on port 6443

### Step 2: Install GPU Operator

The NVIDIA GPU Operator automates the management of NVIDIA software components needed to run GPU workloads in Kubernetes.

```bash
# Run the installation script
./scripts/02-install-gpu-operator.sh

# Or manually:
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace \
  --set driver.enabled=false \      # We already have drivers installed
  --set devicePlugin.enabled=false \ # HAMi will provide this
  --set toolkit.enabled=true         # We need container toolkit

# Wait for toolkit pods to be ready (important!)
kubectl get pods -n gpu-operator -w
# Wait until nvidia-container-toolkit-daemonset pods show Running
```

**What this does:**
- Installs NVIDIA Container Toolkit (nvidia-ctk)
- Configures containerd to use nvidia runtime
- Creates the `nvidia` RuntimeClass
- Does NOT install device plugin (HAMi provides this)

**Why we disable the default device plugin:**
The standard NVIDIA device plugin doesn't support VRAM limits. HAMi's device plugin replaces it with VRAM-aware scheduling.

### Step 3: Install HAMi

HAMi provides the actual GPU virtualization and VRAM enforcement.

```bash
# Run the installation script
./scripts/03-install-hami.sh

# Or manually:
git clone https://github.com/Project-HAMi/HAMi.git ~/HAMi
cd ~/HAMi

# Build helm dependencies
helm dependency build charts/hami

# Create RuntimeClass (if not created by GPU Operator)
kubectl apply -f - <<EOF
apiVersion: node.k8s.io/v1
kind: RuntimeClass
metadata:
  name: nvidia
handler: nvidia
EOF

# Label your GPU node (required for device plugin scheduling)
kubectl label nodes $(hostname) gpu=on

# Install HAMi
helm install hami charts/hami \
  --namespace kube-system \
  --set devicePlugin.deviceMemoryScaling=1 \
  --set devicePlugin.deviceSplitCount=6 \
  --set devicePlugin.runtimeClassName=nvidia

# Verify installation
kubectl get pods -n kube-system | grep hami
# hami-device-plugin-xxxxx   1/1   Running
# hami-scheduler-xxxxx       1/1   Running
```

**What this does:**
- Installs HAMi scheduler (replaces default scheduler for GPU pods)
- Installs HAMi device plugin (exposes `nvidia.com/gpu` and `nvidia.com/gpumem`)
- Configures each GPU to be split into 6 slices

**Key Helm values explained:**

| Value | Default | Description |
|-------|---------|-------------|
| `devicePlugin.deviceSplitCount` | 10 | Max number of slices per GPU |
| `devicePlugin.deviceMemoryScaling` | 1 | Memory overcommit ratio (1 = no overcommit) |
| `devicePlugin.runtimeClassName` | nvidia | RuntimeClass to use |

### Step 4: Verify GPU Resources

```bash
# Check that GPU resources are available
kubectl describe node $(hostname) | grep -A 20 "Allocatable"

# You should see:
# Allocatable:
#   cpu:                8
#   memory:             32Gi
#   nvidia.com/gpu:     12    # 2 GPUs × 6 slices
#   nvidia.com/gpumem:  ...   # Total VRAM
```

---

## Usage

### Create a Student Workspace

```bash
# 1. Create namespace with quota and PVC
./scripts/create-student.sh alice

# 2. Start the GPU workspace
./scripts/start-workspace.sh alice

# 3. Access the workspace
kubectl exec -it -n student-alice workspace -- bash
```

### Verify VRAM Limit

```bash
# Run the test script
./scripts/test-vram-limit.sh alice

# Or manually inside the container:
kubectl exec -it -n student-alice workspace -- bash

# Inside container:
nvidia-smi
# +-----------------------------------------------------------------------------+
# |   0  NVIDIA GeForce RTX 4090   |   0MiB /  8192MiB |      0%      Default |
# +-----------------------------------------------------------------------------+
#                                         ^^^^^ 8GB limit, not 24GB!
```

### Test CUDA Memory Enforcement

```python
# Inside the container:
python3 << 'EOF'
import torch

# Check visible memory
total = torch.cuda.get_device_properties(0).total_memory
print(f"Total VRAM visible: {total / 1e9:.2f} GB")  # Shows ~8GB

# Allocate within limit (works)
x = torch.randn(1000, 1000, 1000, device='cuda')  # ~4GB
print("4GB allocation: SUCCESS")
del x
torch.cuda.empty_cache()

# Allocate over limit (blocked by HAMi)
try:
    y = torch.randn(2000, 2000, 2000, device='cuda')  # ~32GB
except RuntimeError as e:
    print(f"Over-limit allocation: BLOCKED")
    # Error: CUDA out of memory
EOF
```

### Stop Workspace

```bash
./scripts/stop-workspace.sh alice
# Pod is deleted, but PVC data persists
# Next time alice starts, their files are still there
```

---

## VRAM Enforcement Deep Dive

### What Gets Enforced

| CUDA Operation | HAMi Behavior |
|----------------|---------------|
| `cudaMalloc()` | Tracked and limited |
| `cudaMallocManaged()` | Tracked and limited |
| `cuMemAlloc()` | Tracked and limited |
| `cudaGetDeviceProperties()` | Returns fake total memory |
| `nvidia-smi` | Shows limited memory |

### What Does NOT Get Enforced

| Scenario | Behavior |
|----------|----------|
| Host (CPU) memory | Not affected |
| GPU compute time | Not limited (use `nvidia.com/gpucores` for this) |
| GPU memory bandwidth | Shared with other users |
| Direct hardware access | Blocked by container isolation |

### Memory Accounting

HAMi tracks memory at the CUDA API level:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     MEMORY TRACKING EXAMPLE                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Allocation 1: torch.randn(1000,1000,1000).cuda()                      │
│  Size: 1000 × 1000 × 1000 × 4 bytes = 4,000,000,000 bytes ≈ 3.7 GB    │
│                                                                         │
│  HAMi tracking:                                                        │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  GPU_MEMORY_LIMIT: 8192 MiB (8,589,934,592 bytes)              │    │
│  │  Current usage:    3814 MiB                                    │    │
│  │  Available:        4378 MiB                                    │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  Allocation 2: torch.randn(2000,2000,2000).cuda()                      │
│  Size: 2000 × 2000 × 2000 × 4 bytes = 32,000,000,000 bytes ≈ 29.8 GB  │
│                                                                         │
│  HAMi check:                                                           │
│  - Requested: 29.8 GB                                                  │
│  - Available: 4.3 GB                                                   │
│  - Result: CUDA_ERROR_OUT_OF_MEMORY                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## PoC Results

### Test Environment

| Component | Specification |
|-----------|---------------|
| Server | shobdo.ddns.net:3334 |
| GPUs | 2× NVIDIA GeForce RTX 4090 (48GB each) |
| Driver | 565.57.01 |
| CUDA | 12.7 |
| K3s | v1.31.5+k3s1 |
| HAMi | Latest (from GitHub) |

### Verified Results

| Test | Result |
|------|--------|
| Multiple concurrent pods | ✅ 4 pods running simultaneously |
| VRAM limit visibility | ✅ nvidia-smi shows 8192 MiB per pod |
| CUDA memory enforcement | ✅ Over-limit allocations blocked |
| HAMi scheduler working | ✅ Pods distributed across GPU slices |

### Capacity Analysis

```
2 GPUs × 48GB each = 96GB total VRAM

With 8GB slices:
  96GB ÷ 8GB = 12 concurrent users

With 4GB slices:
  96GB ÷ 4GB = 24 concurrent users

With mixed allocation:
  6 × 8GB (standard) = 48GB
  12 × 4GB (free tier) = 48GB
  Total: 18 concurrent users
```

---

## Configuration

### Change VRAM Allocation

Edit `templates/workspace-pod.yaml`:

```yaml
resources:
  limits:
    nvidia.com/gpumem: 4096    # 4GB instead of 8GB
```

### Change Slices Per GPU

```bash
helm upgrade hami ~/HAMi/charts/hami \
  --namespace kube-system \
  --set devicePlugin.deviceSplitCount=12  # 12 slices per GPU
```

### Enable GPU Compute Limiting

```yaml
resources:
  limits:
    nvidia.com/gpu: 1
    nvidia.com/gpumem: 8192
    nvidia.com/gpucores: 50    # Limit to 50% of GPU compute
```

### Workspace Pod Customization

The workspace pod can be customized for different use cases:

```yaml
# For JupyterLab
containers:
- name: workspace
  image: jupyter/pytorch-notebook:latest
  ports:
  - containerPort: 8888

# For VS Code Server
containers:
- name: workspace
  image: codercom/code-server:latest
  ports:
  - containerPort: 8080
```

---

## Troubleshooting

See [docs/troubleshooting.md](docs/troubleshooting.md) for detailed solutions.

### Quick Diagnostics

```bash
# Check all GPU-related pods
kubectl get pods -A | grep -E "hami|nvidia|gpu"

# Check HAMi logs
kubectl logs -n kube-system -l app=hami-device-plugin --tail=50
kubectl logs -n kube-system -l app=hami-scheduler --tail=50

# Check GPU resources on node
kubectl describe node $(hostname) | grep nvidia.com

# Debug a pending pod
kubectl describe pod <pod-name> -n <namespace>
```

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Pod stuck in Pending | No GPU resources | Check HAMi device plugin is running |
| nvidia-smi shows full VRAM | Missing schedulerName | Add `schedulerName: hami-scheduler` |
| Container can't find libcuda | Missing runtimeClassName | Add `runtimeClassName: nvidia` |
| HAMi device plugin crash | Toolkit not ready | Install GPU Operator first |

---

## Limitations

### Current Limitations

1. **No GPU compute isolation** - Users share GPU compute cycles (can add `nvidia.com/gpucores` for partial mitigation)
2. **No network bandwidth isolation** - GPU-to-GPU or GPU-to-CPU bandwidth is shared
3. **Single-GPU per pod** - HAMi doesn't support multi-GPU pods with partial VRAM
4. **No live migration** - Can't move running workloads between GPUs

### HAMi vs Hardware Virtualization

| Feature | HAMi | NVIDIA MIG | SR-IOV |
|---------|------|------------|--------|
| VRAM isolation | Software | Hardware | Hardware |
| Compute isolation | Partial | Full | Full |
| Fault isolation | No | Yes | Yes |
| Supported GPUs | All NVIDIA | A100/H100 | Limited |
| Performance overhead | ~1-2% | ~0% | ~0% |

---

## References

- [HAMi GitHub Repository](https://github.com/Project-HAMi/HAMi)
- [HAMi Documentation](https://project-hami.io/)
- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/)
- [K3s Documentation](https://docs.k3s.io/)
- [Kubernetes Device Plugins](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/)

---

## License

MIT
