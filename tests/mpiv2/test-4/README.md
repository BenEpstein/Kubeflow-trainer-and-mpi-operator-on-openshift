# Test-4: PyTorch DDP MPI Training Tests

## Overview

This directory contains comprehensive tests for running distributed PyTorch Distributed Data Parallel (DDP) training on the MNIST dataset using MPI Operator v2 in OpenShift/Kubernetes 4.12.

## Goal

The primary goal of this test suite is to validate distributed PyTorch DDP training capabilities using:
	•	PyTorch DDP: Native distributed training framework built into PyTorch
	•	MPI (Message Passing Interface): Inter-process communication for multi-pod training
	•	MPI Operator v2: Kubeflo MPI job controller
	•	OpenShift Security Context Constraints (SCC): Enforces non-root, secure pod execution

## Test Scenario

This test uses the following configuration:
	•	GPU-based Training (torch-mnist-ddp-mpijob.yaml)
	•	Each worker pod requests 1 GPU (nvidia.com/gpu: 1)
	•	Uses NCCL backend for multi-GPU communication (falls back to Gloo on CPU)
	•	Includes extra debug flags for PyTorch DDP and NCCL stability over Ethernet

## Directory Structure

```
test-4/
── README.md                        # This documentation file
├── torch-mnist-ddp-mpijob.yaml     # PyTorch DDP MPI job configuration
├── scc.yaml                        # Security Context Constraints for OpenShift
├── rolebinding.yaml                # RBAC configuration
└── image/                          # Container image build files
    ├── Dockerfile                  # Container image for training
    ├── torch_mnist_ddp.py          # PyTorch DDP MNIST training script
    └── sshd_config                 # SSH daemon configuration
```

## How to Run the Test

Prerequisites
	1.	MPI Operator v2 installed on your cluster
	2.	RBAC permissions configured for your service account
	3.	NVIDIA GPU Operator (or equivalent) for GPU support
	4.	OpenShift cluster (for SCC support) or equivalent security setup

## Running the Test

**Deploy all resources in the test-4 directory:**

```bash
# Deploy all test resources
oc apply -f test-4/

# Or apply individual components:
oc apply -f test-4/scc.yaml
oc apply -f test-4/rolebinding.yaml
oc apply -f test-4/torch-mnist-ddp-mpijob.yaml
```


**Monitoring Execution**

```bash
# Check MPI job status
oc get mpijob torch-ddp-mnist

# Inspect pod creation
oc get po

# Follow launcher logs (training output)
oc logs -f <launcher-pod-name>

# Debug pod issues
oc describe pod <pod-name>
```

**Cleanup**

```bash
# Remove all test resources
oc delete -f test-4/

# Or delete specific resources
oc delete mpijob torch-ddp-mnist
```

## Dockerfile Explanation

The image/Dockerfile builds a container optimized for PyTorch DDP with MPI:

**Base Image**

```bash
FROM nvcr.io/nvidia/pytorch:24.05-py3
```

	•	NVIDIA’s official PyTorch container
	•	Includes CUDA, cuDNN, NCCL, and HPC-X Open MPI

Key Features

	•	Non-root user (mpiuser, UID 10001) for OpenShift SCC compliance
	•	HPC-X Open MPI pre-bundled in the base image
	•	Installed tools: openssh-server, mpi4py, torchvision
	•	SSHD setup: Custom config on port 2222 for MPI Operator connectivity
	•	Tini init process: Ensures clean process reaping inside container

**PyTorch Training Script (torch_mnist_ddp.py)**

Features

	•	Distributed setup: Initializes PyTorch process groups using nccl (GPU) or gloo (CPU)
	•	DDP model wrapping: Ensures gradient synchronization across workers
	•	Distributed samplers: Ensures each rank processes unique dataset partitions
	•	Metrics aggregation: Uses torch.distributed.all_reduce for test loss and accuracy
	•	Debug info: Prints rank, world size, hostname, and CUDA availability

Training Flow

	1.	Initialize distributed environment (init_distributed())
	2.	Load MNIST dataset with DistributedSampler
	3.	Train for 3 epochs with Adam optimizer
	4.	Evaluate accuracy across all workers (synchronized metrics)
	5.	Destroy process group and exit cleanly

**Security Context Constraints (SCC)**

The scc.yaml is the same as used in Test-3 (TensorFlow Horovod):

	•	Runs as non-root (runAsUser: 10001)
	•	Disables privilege escalation
	•	Drops all Linux capabilities
	•	Allows necessary inter-pod communication for MPI and NCCL
	•	Provides writable filesystem for SSH and temporary dataset storage

