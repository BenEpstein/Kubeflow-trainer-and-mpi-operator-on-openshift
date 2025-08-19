# Test 1: Multi-node Training with Privileged SCC

Description
- Deploy a running multi-node training job using the MPI Operator and kubeflow-trainer on OpenShift 4.12.
- Requires the Privileged SCC to allow the pods the necessary permissions.

How to run
- Ensure you are on the correct project/namespace (oc project <namespace>).
- Apply the manifests in this directory:

oc apply -f tests/mpiv2/test-1

Notes
- Confirm the Privileged SCC (or appropriate permissions) is granted to the service account used by this test.
- Check pod status and logs to verify multi-node scheduling and training progress.
