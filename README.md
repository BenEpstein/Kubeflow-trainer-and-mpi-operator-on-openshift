# Distributed Training: Kubeflow Trainer on OpenShift 4.12

This directory is a testing workspace for deploying and validating the MPI Operator and kubeflow-trainer on OpenShift 4.12. It contains Kustomize bases and overlays for operators and example tests you can apply to exercise the stack.

## Directory Structure

- base/
  Kustomize bases for core components (e.g., CRDs, operator subscriptions, RBAC). These are environment-agnostic building blocks that overlays compose.

- overlays/
  Environment- or cluster-specific Kustomize overlays that reference the bases and apply patches, namespaces, and settings for a particular cluster.
  - overlays/<cluster>/
    A concrete overlay for a target cluster (names, labels, image registries, and other cluster-specific settings go here).

- tests/
  Self-contained examples that run against a target namespace to validate deployments and flows.
  - tests/<framework>/<test>/
    A framework-specific test (for example, mpiv2/test-1). Each test directory may include a namespace.yaml (or assume an existing namespace) and manifests/jobs to run.

## How to Deploy Operators (MPI Operator, Kubeflow Trainer)

Use your cluster-specific overlay with server-side apply:

```bash
$ oc apply --server-side -k overlays/<cluster>
```


Replace <cluster> with the desired overlay directory name under overlays/.

## How to Run Tests

Apply a test manifest directory directly:

```bash
$ oc apply -f tests/<framework>/<test>
```

Examples:

```bash
$ oc apply -f tests/mpiv2/test-1
```

## Notes

- Ensure your oc context points to the correct OpenShift 4.12 cluster and project (namespace) before applying manifests (oc project <name>).
- Tests are designed to be minimal and focused; feel free to duplicate a test directory to create new scenarios.
- Keep cluster-specific changes in overlays/ and reusable resources in base/.

