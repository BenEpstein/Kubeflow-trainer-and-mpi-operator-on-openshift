# CUDA + cuDNN runtime base (example)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG SSH_PORT=2222
ENV DEBIAN_FRONTEND=noninteractive \
    HOME=/home/mpiuser \
    USER=mpiuser \
    SHELL=/bin/bash

# OS + SSH + Open MPI + Python
RUN apt-get update && apt-get install -y --no-install-recommends \
      openssh-server openssh-client openmpi-bin libopenmpi-dev \
      python3 python3-pip ca-certificates tini \
  && rm -rf /var/lib/apt/lists/*

# Non-root user with stable UID/GID for SCC
RUN groupadd -g 1001 mpiuser && useradd -m -u 1001 -g 1001 -s /bin/bash mpiuser

# Python deps
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
 && python3 -m pip install --no-cache-dir mpi4py

USER 1001
WORKDIR /home/mpiuser

# User-scoped sshd config. HostKey + AuthorizedKeysFile read directly from Secret at /src
RUN mkdir -p ${HOME}/.ssh ${HOME}/bin
RUN cat > ${HOME}/.sshd_config <<EOF
Port ${SSH_PORT}
PidFile /run/sshd/sshd.pid
HostKey /src/ssh-privatekey
StrictModes no
PasswordAuthentication no
PubkeyAuthentication yes
AuthorizedKeysFile /src/ssh-publickey
UsePAM no
Subsystem sftp internal-sftp
EOF

# Client config: do not prompt, use custom port
RUN printf "Host *\n  Port %s\n  StrictHostKeyChecking no\n  UserKnownHostsFile /dev/null\n" "${SSH_PORT}" > ${HOME}/.ssh/config

# Helper to run sshd in foreground
RUN cat > ${HOME}/bin/start-sshd.sh <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
test -d /run/sshd || { echo "/run/sshd missing"; exit 1; }
exec /usr/sbin/sshd -D -f "$HOME/.sshd_config"
EOS
RUN chmod +x ${HOME}/bin/start-sshd.sh

EXPOSE ${SSH_PORT}
ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["bash","-lc","sleep infinity"]
