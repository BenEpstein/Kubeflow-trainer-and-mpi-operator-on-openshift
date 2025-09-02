# Test-3: TensorFlow Horovod MPI Training Tests

## Overview

This directory contains comprehensive tests for running distributed TensorFlow training using Horovod with MPI on OpenShift/Kubernetes using the MPI Operator v2.

### Goal

The primary goal of this test suite is to validate distributed TensorFlow training capabilities using:
- **Horovod**: A distributed deep learning training framework
- **MPI (Message Passing Interface)**: For inter-process communication
- **MPI Operator v2**: Kubeflow's MPI job controller
- **OpenShift Security Context Constraints (SCC)**: For secure pod execution

### Test Scenarios

This directory includes two main test configurations:

1. **GPU-based Training** (`tf-mnist-hvd-mpijob.yaml`)
   - Distributed training with GPU acceleration
   - Each pod requests 1 GPU (`nvidia.com/gpu: 1`)
   - Optimized for high-performance GPU clusters

2. **CPU-based Training** (`tf-mnist-hvd-mpijob-cpu.yaml`)
   - Distributed training using CPU resources only
   - Resource requests: 1 CPU core, 2GiB memory per pod
   - Resource limits: 2 CPU cores, 4GiB memory per pod
   - Suitable for environments without GPU resources

### Directory Structure

```
test-3/
── README.md                      # This documentation file
├── tf-mnist-hvd-mpijob.yaml      # GPU-based MPI job configuration
├── tf-mnist-hvd-mpijob-cpu.yaml  # CPU-based MPI job configuration
├── scc.yaml                      # Security Context Constraints for OpenShift
├── rolebinding.yaml              # RBAC configuration
└── image/                        # Container image build files
    ├── Dockerfile                # Multi-stage container build
    ├── tf_mnist_hvd.py          # TensorFlow MNIST training script
    └── sshd_config              # SSH daemon configuration
```

## How to Run the Tests

### Prerequisites

1. **MPI Operator v2** must be installed in your cluster
2. **Appropriate RBAC permissions** for your service account
3. **GPU support** (for GPU test) - NVIDIA GPU Operator or equivalent
4. **OpenShift cluster** (for SCC support) or Kubernetes with equivalent security policies

### Running the Tests

Apply all configurations in the test-3 directory:

```bash
# Deploy all test resources at once
oc apply -f test-3/

# Or apply individual components:
oc apply -f test-3/scc.yaml
oc apply -f test-3/rolebinding.yaml

# Run GPU-based test
oc apply -f test-3/tf-mnist-hvd-mpijob.yaml

# Or run CPU-based test
oc apply -f test-3/tf-mnist-hvd-mpijob-cpu.yaml
```

### Monitoring Test Execution

```bash
# Check MPI job status
$ oc get mpijob <>

# If pods aren't creating check for SCC errors
$ oc describe mpijob <> 

# Monitor pod creation and status
oc get po

# View training logs
oc logs -f <launcher-pod-name>

# Check detailed pod events
oc describe pod <pod-name>
```

### Cleanup

```bash
# Clean up all test resources
oc delete -f test-3/

# Or clean up specific components
oc delete mpijob tf-horovod-mnist
oc delete mpijob tf-horovod-mnist-cpu
```

## Dockerfile Explanation

The `image/Dockerfile` creates a secure, optimized container for distributed TensorFlow training:

### Base Image
```dockerfile
FROM nvcr.io/nvidia/tensorflow:24.05-tf2-py3
```
- Uses NVIDIA's official TensorFlow container with CUDA support
- Includes TensorFlow 2.x with GPU acceleration capabilities
- Pre-configured with NVIDIA libraries and drivers

### Security Hardening
```dockerfile
ARG USER=mpiuser
ARG UID=10001
```
- Creates a **non-root user** (`mpiuser`) with UID 10001
- Follows OpenShift security best practices
- Prevents privilege escalation attacks

### Network Resilience
```dockerfile
RUN echo 'Acquire::ForceIPv4 "true";' > /etc/apt/apt.conf.d/99force-ipv4
```
- Forces IPv4 for apt operations to avoid networking issues
- Adds retry mechanisms for package installations
- Switches HTTP sources to HTTPS for proxy compatibility

### MPI and SSH Configuration
```dockerfile
RUN apt-get install -y openssh-server openssh-client openmpi-bin libopenmpi-dev
```
- Installs **OpenSSH** for inter-pod communication
- Installs **Open MPI** libraries for distributed computing
- Configures SSH to work in containerized environments without host key checking

### Horovod Installation
```dockerfile
RUN pip install --no-cache-dir horovod[tensorflow,mpi]
```
- Installs Horovod with TensorFlow and MPI support
- Enables distributed training across multiple nodes/pods

### Custom SSH Configuration
- Uses custom `sshd_config` for non-root SSH daemon
- Configured to run on port 2222 (non-privileged port)
- Optimized for MPI Operator communication patterns

## Security Context Constraints (SCC) Explanation

The `scc.yaml` file defines OpenShift Security Context Constraints that govern how pods can run:

### Key Security Settings

```yaml
allowPrivilegeEscalation: false
allowPrivilegedContainer: false
```
- **Prevents privilege escalation** - pods cannot gain additional privileges
- **Blocks privileged containers** - enhanced security posture

### User Security
```yaml
runAsUser:
  type: MustRunAs
  uid: 10001
```
- **Enforces non-root execution** - all containers must run as UID 10001
- **Consistent user identity** - matches the Dockerfile's mpiuser

### Inter-Pod Communication
```yaml
allowHostIPC: true
```
- **Enables Host IPC** - required for MPI communication between pods
- **Allows shared memory** - critical for high-performance computing workloads

### Volume and Capability Management
```yaml
allowedCapabilities: []
requiredDropCapabilities: null
volumes: ['*']
```
- **Minimal capabilities** - no additional Linux capabilities granted
- **Flexible volume support** - allows necessary volume mounts for SSH keys and data
- **Default capability dropping** - removes unnecessary privileges

### File System Security
```yaml
fsGroup:
  type: RunAsAny
readOnlyRootFilesystem: false
```
- **Flexible file system groups** - accommodates MPI shared file requirements
- **Writable root filesystem** - necessary for SSH and MPI operations
