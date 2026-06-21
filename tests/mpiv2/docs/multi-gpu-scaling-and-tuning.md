# Scaling MPIJobs to Multiple GPUs per Worker (2×2) + Resource Tuning

A reference for running the Horovod/TensorFlow (`test-3`) and PyTorch DDP (`test-4`)
MPIJobs with **2 worker replicas, each holding 2 GPUs** (4 ranks total) on
OpenShift, using MPI Operator v2 (`mpioperator/mpi-operator:0.6.0`).

It covers how the jobs are wired, how `/etc/mpi/hostfile` is produced, the
thread‑exhaustion crash seen at 2×2 in a disconnected cluster, and how to make
the launch parameters scale automatically from the hostfile.

---

## 1. TL;DR — does 2×2 work today?

| Test | As written today | Works for 2 workers × 2 GPUs (4 ranks)? |
|------|------------------|------------------------------------------|
| **test-4 — PyTorch DDP** (`torch-ddp-mnist`) | `slotsPerWorker: 2`, `nvidia.com/gpu: 2`, `mpirun -np ${WORLD_SIZE}` derived from the hostfile | ✅ Already configured for exactly this |
| **test-3 — Horovod/TF** (`tf-horovod-mnist`) | `slotsPerWorker: 1`, `nvidia.com/gpu: 1`, **`mpirun -np 2` hardcoded** | ⚠️ Not without edits (would run 1 rank/worker, second GPU idle) |

**Both Python scripts are already topology-agnostic** — they pin one GPU per
*local* rank and shard/scale by *total* size, so no code change is needed. Only
the **manifests** differ: test-4 scales dynamically; test-3 is pinned to the
"2 GPUs total" case.

To convert test-3 to 2×2, three values must agree:

```yaml
spec:
  slotsPerWorker: 2          # was 1   → hostfile advertises slots=2
# launcher:
  mpirun -np 4 ...           # was 2   → 2 workers × 2 slots  (better: derive from hostfile)
# worker resources:
  requests: { nvidia.com/gpu: 2 }   # was 1
  limits:   { nvidia.com/gpu: 2 }   # was 1
```

> Note: the **test-4 README** says "1 GPU per worker", but its YAML actually
> requests `2` with `slotsPerWorker: 2`. The YAML is the source of truth.

---

## 2. How an MPIJob runs (roles + exec flow)

The operator splits the job into two roles:

- **Worker pods** (`<job>-worker-0`, `<job>-worker-1`): their entire command is
  `exec /usr/sbin/sshd -D` on port **2222**. They hold the GPUs and run the
  actual training ranks. They do no orchestration themselves.
- **Launcher pod**: runs `mpirun`. `mpirun` is *not* an MPI rank — it's the
  orchestrator. It SSHes into each worker (Open MPI's default `rsh` agent is
  `ssh`), starts `orted`, and spawns the `-np N` ranks **on the workers**.

Consequences:
- **GPUs are requested on the workers, not the launcher** (that's where compute runs).
- `launcherCreationPolicy: WaitForWorkersReady` holds the launcher until every
  worker's sshd is up, avoiding "connection refused" races.
- The operator mounts an SSH keypair (Secret) at `sshAuthMountPath`
  (`/home/mpiuser/.ssh`): private key on the launcher, `authorized_keys` on the
  workers. Images set ssh/sshd to port 2222.

Rank placement is driven by **`-np` (how many ranks) + the hostfile's `slots=`
(how many ranks per host)** — both must match the GPU requests.

---

## 3. How `/etc/mpi/hostfile` is populated

You never write this file — the **mpi-operator generates it**:

1. **ConfigMap.** The controller creates `<job>-config` with keys `hostfile` and
   `discover_hosts.sh`, mounted into the launcher at `/etc/mpi`.
2. **Content** (default `mpiImplementation: OpenMPI`): one line per worker,
   using `spec.slotsPerWorker` (CRD default `1`):
   ```
   <job>-worker-<i>.<job>-worker.<namespace>.svc slots=<slotsPerWorker>
   ```
3. **DNS.** A headless Service `<job>-worker` makes those FQDNs resolve to
   individual worker pods.
4. **Auto-read by mpirun.** The operator sets
   `OMPI_MCA_orte_default_hostfile=/etc/mpi/hostfile` (and
   `OMPI_MCA_orte_keep_fqdn_hostnames=true`) on the launcher — which is why the
   manifests run `mpirun -np N` with **no `-hostfile` flag**.
5. **Updates** if you scale the worker count.

Concrete result for test-4 at 2×2 (namespace `mpi-demo`):
```
torch-ddp-mnist-worker-0.torch-ddp-mnist-worker.mpi-demo.svc slots=2
torch-ddp-mnist-worker-1.torch-ddp-mnist-worker.mpi-demo.svc slots=2
```

> For Intel/MPICH the line format is `host:slots` and a different env var is used.

---

## 4. Deriving WORLD_SIZE from the hostfile

test-4's launcher computes total ranks by summing `slots=` across the hostfile
(POSIX awk — runs under the NGC image's `mawk`):

```bash
WORLD_SIZE=$(awk '{c=0;for(i=1;i<=NF;i++)if($i~/^slots=/){split($i,a,"=");c=a[2]};if(c==0)c=1;t+=c}END{print t}' /etc/mpi/hostfile)
```

Verified: yields **4** for 2 workers × 2 slots, **2** for 2 workers × 1 slot.

Notes:
- For **PyTorch** this value feeds both `mpirun -np` *and* the script's
  `WORLD_SIZE` env (DDP reads it).
- For **Horovod** you only need it for `-np`; Horovod reads world size from MPI
  (`hvd.size()` ← `OMPI_COMM_WORLD_SIZE`), not a `WORLD_SIZE` env var.
- **Caveat:** the `^slots=` match assumes the **OpenMPI** hostfile format. With
  Intel/MPICH (`host:N`) it would fall back to 1/host and undercount.

---

## 5. The 2×2 crash (disconnected cluster, Horovod)

Observed errors:
```
tensorflow/tsl/platform/default/subprocess.cc:304] start cannot spawn child process: No such file or directory
Check failed: ret == 0 (11 vs. 0) Thread tf_creation via pthread_create() failed
Signal aborted (-6)
```

**These are one failure, not three:**

| Line | Role |
|------|------|
| `Check failed … pthread_create() failed` (ret **11 = EAGAIN**) | **THE FATAL ONE** — thread creation refused due to a resource limit. `pthread_create` returns the errno as its value; 11 = EAGAIN. This is a `LOG(FATAL)`. |
| `Signal aborted (-6)` = SIGABRT(6) | The `abort()` raised *by* that fatal CHECK. A symptom. |
| `subprocess.cc … cannot spawn child process` | A failed `fork()`/`exec()` from the same exhaustion. Collateral, not a missing binary. |

### Root cause
Going from **1 rank/pod → 2 ranks/pod** doubles thread/PID pressure in one
cgroup. TensorFlow (and OpenMP/Eigen/MKL/CUDA/NCCL) size their thread pools to
the **node's full core count** (`sysconf(_SC_NPROCESSORS_ONLN)`), ignoring the
cgroup CPU quota. Two such processes in one pod cross one of these ceilings:

1. **cgroup `pids.max`** (a default of 1024 is common).
2. **`RLIMIT_NPROC` (`ulimit -u`)** — counted **per real UID across the whole
   node**, not per container. Every pod here runs as the same fixed UID `10001`
   (SCC `MustRunAs`), so co-scheduled pods share the count. Major aggravator.
3. **`vm.max_map_count`** (default 65530) — each thread stack is an mmap.
4. **Pod memory** for thread stacks / two CUDA contexts.

It didn't show at 2×1 because one process per pod stayed under the ceiling.

### Confirm which ceiling (inside a worker pod)
```bash
ulimit -a                      # "-u" max user processes, "-v", "-s" stack
cat /sys/fs/cgroup/pids.max    # and pids.current (cgroup v2)
cat /proc/sys/vm/max_map_count
nproc                          # vs CPU limit — if huge, thread oversubscription is it
ps -eLf | wc -l                # live thread count
```

### Important distinction: 4 ranks ≠ the problem
The **4 MPI ranks are correct and stay 4.** The crash is about the *threads
created inside each rank*. The fix caps per-rank threads; it does not change the
rank/GPU count or parallelism across GPUs.

| Term | Count | Controlled by | Fix touches it? |
|------|-------|---------------|-----------------|
| MPI ranks (OS processes, one python/GPU) | 4 | `-np` / `slotsPerWorker` | ❌ stays 4 |
| Threads inside each rank | hundreds | `OMP_NUM_THREADS`, `TF_NUM_*`, … | ✅ capped |

---

## 6. TensorFlow vs PyTorch susceptibility

Same class of problem — the kernel/cgroup limits are framework-agnostic.
Differences in *likelihood*:

- PyTorch's `libtorch`/ATen also sizes pools to node cores → same risk.
- PyTorch is usually a bit leaner (TF adds extra Eigen + inter-op pools), and
  test-4's `DataLoader(num_workers=0)` adds **no** data-loader subprocesses.
- **Key nuance:** PyTorch's own launchers (`torchrun`) auto-set
  `OMP_NUM_THREADS=1` for multi-process-per-node. Launching via **`mpirun`
  (MPI Operator) bypasses that guard**, so test-4 is just as exposed in principle.
- Different signatures to watch for:
  - `RuntimeError: Resource temporarily unavailable` / `std::system_error` →
    same EAGAIN thread/PID exhaustion → SIGABRT(6).
  - `Bus error` / **SIGBUS(7)** or `NCCL WARN … unhandled system error` → that's
    the `/dev/shm` ceiling, not threads.
  - `DataLoader worker (pid …) exited unexpectedly … [Errno 11]` → only if you
    raise `num_workers`.

---

## 7. The fix — four thread parameters

Each caps a **separate** thread pool that defaults to node-core-count. On a GPU
job these CPU pools are mostly idle but still *get created*; (pools × ranks/pod)
is what exhausts the PID/thread budget.

| Parameter | Controls | Why cap it |
|-----------|----------|------------|
| **OMP_NUM_THREADS** | OpenMP pool — master CPU-parallelism knob used by Eigen (TF), ATen (PyTorch), BLAS. | Stops ~1 thread/core per rank; most libs honor it as the default. |
| **TF_NUM_INTRAOP_THREADS** | Parallelism *inside* one op (split a matmul/conv across threads). Default = core count. | Bounds threads-per-op per rank. Set ≈ cpus/slots. |
| **TF_NUM_INTEROP_THREADS** | Parallelism *across independent* ops (run two graph branches at once). Default = core count. | Rarely needs > ~2; avoids a second full-size pool. |
| **OPENBLAS_NUM_THREADS / MKL_NUM_THREADS** | The BLAS/LAPACK library's *own* internal pool for CPU matrix math (OpenBLAS / Intel MKL). | These nest *inside* OMP/TF ops, so uncapped they multiply (op-threads × BLAS-threads). Pin low (1). |

> PyTorch uses `OMP_NUM_THREADS` + `MKL_NUM_THREADS`; the two `TF_*` vars are
> TensorFlow-specific and harmless to pass to Torch.

### Automatic sizing from the hostfile
`threads-per-rank = CPUs-per-worker ÷ slots`. `slots` comes from the hostfile
automatically; `CPUS_PER_WORKER` is the one value you supply (match it to the
worker container's `limits.cpu`). This adapts whenever slots/GPUs change.

With `CPUS_PER_WORKER=8`, `slots=2` → `OMP=4` (the known-good value); `slots=4`
→ `2`; `slots=1` → `8` — no edits.

> Fully zero-input alternative: compute per rank on the **worker** side at exec
> time using `OMPI_COMM_WORLD_LOCAL_SIZE` (= the hostfile `slots` for that pod)
> ÷ the pod's own cgroup CPU quota. Requires a small entry wrapper rather than an
> `-x` flag, because `-x` can't compute on the remote rank.

---

## 8. The one-liner (launcher command)

Single `mpirun` command, every value computed inline from `/etc/mpi/hostfile`
(like `-np`), no helper variables:

```bash
mpirun -np $(awk '{c=0;for(i=1;i<=NF;i++)if($i~/^slots=/){split($i,a,"=");c=a[2]};if(c==0)c=1;t+=c}END{print t}' /etc/mpi/hostfile) -bind-to none -map-by slot -x OMP_NUM_THREADS=$(awk -v cpw=${CPUS_PER_WORKER:-8} 'NR==1{for(i=1;i<=NF;i++)if($i~/^slots=/){split($i,a,"=");s=a[2]};if(s<1)s=1;o=int(cpw/s);print(o<1?1:o);exit}' /etc/mpi/hostfile) -x TF_NUM_INTRAOP_THREADS=$(awk -v cpw=${CPUS_PER_WORKER:-8} 'NR==1{for(i=1;i<=NF;i++)if($i~/^slots=/){split($i,a,"=");s=a[2]};if(s<1)s=1;o=int(cpw/s);print(o<1?1:o);exit}' /etc/mpi/hostfile) -x TF_NUM_INTEROP_THREADS=2 -x OPENBLAS_NUM_THREADS=1 -x MKL_NUM_THREADS=$(awk -v cpw=${CPUS_PER_WORKER:-8} 'NR==1{for(i=1;i<=NF;i++)if($i~/^slots=/){split($i,a,"=");s=a[2]};if(s<1)s=1;o=int(cpw/s);print(o<1?1:o);exit}' /etc/mpi/hostfile) -x NCCL_DEBUG=INFO -x HOROVOD_LOG_LEVEL=INFO -x LD_LIBRARY_PATH -x PATH python /workspace/tf_mnist_hvd.py
```

Compute-once variant (still one physical line, runs the OMP awk just once):

```bash
WORLD_SIZE=$(awk '{c=0;for(i=1;i<=NF;i++)if($i~/^slots=/){split($i,a,"=");c=a[2]};if(c==0)c=1;t+=c}END{print t}' /etc/mpi/hostfile); SLOTS=$(awk 'NR==1{for(i=1;i<=NF;i++)if($i~/^slots=/){split($i,a,"=");print a[2];exit}}' /etc/mpi/hostfile); SLOTS=${SLOTS:-1}; OMP=$(( ${CPUS_PER_WORKER:-8} / SLOTS )); [ "$OMP" -ge 1 ] || OMP=1; mpirun -np ${WORLD_SIZE} -bind-to none -map-by slot -x OMP_NUM_THREADS=${OMP} -x TF_NUM_INTRAOP_THREADS=${OMP} -x TF_NUM_INTEROP_THREADS=2 -x OPENBLAS_NUM_THREADS=1 -x MKL_NUM_THREADS=${OMP} -x NCCL_DEBUG=INFO -x HOROVOD_LOG_LEVEL=INFO -x LD_LIBRARY_PATH -x PATH python /workspace/tf_mnist_hvd.py
```

In YAML, use a `|` block scalar so the awk's `"="` and `'…'` quotes don't fight YAML:

```yaml
command: ["/bin/bash","-lc"]
args:
  - |
    set -euo pipefail
    mpirun -np $(awk '...') ... python /workspace/tf_mnist_hvd.py
```

For PyTorch, swap the target to `python /workspace/torch_mnist_ddp.py` (keep the
existing per-rank `RANK`/`LOCAL_RANK` wiring).

---

## 9. The four `mpirun` flags

| Flag | Purpose |
|------|---------|
| **`-np 4`** | How many ranks to launch (2 workers × 2 slots). *Where* they go is read from the hostfile. |
| **`-bind-to none`** | CPU-affinity policy. Default pins each rank to cores; `none` lets each rank float across the pod's allowed CPUs. Right for GPU work + multiple ranks/pod + cgroup CPU limits (binding inside a fractional cgroup can mis-pin or fail). |
| **`-map-by slot`** | Placement: fill a host's slots before the next → ranks 0,1 on worker-0, 2,3 on worker-1. Makes `OMPI_COMM_WORLD_LOCAL_RANK` = 0,1 per pod (→ one GPU each) and puts rank 0 on the first host (→ `MASTER_ADDR`). `-map-by node` would round-robin one rank/host instead. |
| **`-x VAR`** | Forward an env var from launcher into every remote rank (ranks run in separate pods over SSH, so don't inherit it). `-x NAME` forwards the current value; `-x NAME=value` sets it explicitly. |

---

## 10. Other resource ceilings (not hostfile-derived — set once)

- **`/dev/shm`** (NCCL intra-node between the 2 GPUs). K8s default is 64 MB:
  ```yaml
  volumes:
    - name: dshm
      emptyDir: { medium: Memory, sizeLimit: 1Gi }
  # worker container:
  volumeMounts:
    - { name: dshm, mountPath: /dev/shm }
  ```
- **`pids.max`** — kubelet `podPidsLimit` / `--pod-max-pids` (node-level).
- **`vm.max_map_count`** — node sysctl or pod `securityContext.sysctls`.
- **CPU `limits`** on workers — required for the cgroup-based thread sizing to
  read a real quota (otherwise it falls back to node cores). Keep equal to
  `CPUS_PER_WORKER`.

---

## 11. Disconnected-environment note (separate from the crash)

Both training scripts pull MNIST from the internet:
- `tf_mnist_hvd.py` → `tf.keras.datasets.mnist.load_data()` (downloads from
  `storage.googleapis.com`).
- `torch_mnist_ddp.py` → `datasets.MNIST(..., download=True)`.

Air-gapped these fail with a URL/connection error — a **different** failure you
hit *after* fixing threads. **Pre-bake the dataset into the image** (or mount it)
and point the loaders at the local path (`KERAS_HOME`/`~/.keras/datasets` for TF;
the dataset root for Torch).

---

## 12. Checklist: convert a job to 2 workers × 2 GPUs

- [ ] `spec.slotsPerWorker: 2`
- [ ] Worker `requests/limits.nvidia.com/gpu: 2`
- [ ] Worker `requests/limits.cpu` set (e.g. 8) and `CPUS_PER_WORKER` launcher env to match
- [ ] `mpirun -np` derived from the hostfile (don't hardcode)
- [ ] Thread caps via `-x` (auto-sized: `cpus/slots`)
- [ ] `/dev/shm` Memory volume mounted on workers
- [ ] Node `pids.max` / `vm.max_map_count` headroom (or anti-affinity so same-UID pods don't share a node)
- [ ] Dataset pre-baked for disconnected clusters
- [ ] No training-script changes required (scripts are topology-agnostic)
