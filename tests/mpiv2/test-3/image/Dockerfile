# Dockerfile
FROM nvcr.io/nvidia/tensorflow:24.05-tf2-py3

# Create a non-root user
ARG USER=mpiuser
ARG UID=10001
ARG port=2222

# Make apt resilient in locked-down networks:
# - Force IPv4 to dodge IPv6 routing issues you saw in logs
# - Retry a few times
# - Switch any http sources to https (many proxies block port 80)
RUN echo 'Acquire::ForceIPv4 "true";' > /etc/apt/apt.conf.d/99force-ipv4 \
 && echo 'Acquire::Retries "6";'      > /etc/apt/apt.conf.d/80retries \
 && sed -i 's|http://|https://|g' /etc/apt/sources.list

# Install OpenSSH server and Open MPI + build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-server openssh-client \
        openmpi-bin libopenmpi-dev \
        python3-pip ca-certificates curl vim tini libcap2-bin \
    && rm -rf /var/lib/apt/lists/*

# Horovod (uses MPI controller; NCCL on GPUs if present)
# The base image already has TF and CUDA libs.
RUN pip install --no-cache-dir horovod[tensorflow,mpi]

# Add capability to run sshd as non-root.
RUN setcap CAP_NET_BIND_SERVICE=+eip /usr/sbin/sshd

# Allow OpenSSH to talk to containers without asking for confirmation
# by disabling StrictHostKeyChecking.
# mpi-operator mounts the .ssh folder from a Secret. For that to work, we need
# to disable UserKnownHostsFile to avoid write permissions.
# Disabling StrictModes avoids directory and files read permission checks.
RUN mkdir -p /var/run/sshd /etc/ssh \
 && sed -i "s/[ #]\(.*StrictHostKeyChecking \).*/ \1no/g" /etc/ssh/ssh_config \
 && echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config \
 && sed -i "s/[ #]\(.*Port \).*/ \1$port/g" /etc/ssh/ssh_config \
 && sed -i "s/#\(StrictModes \).*/\1no/g" /etc/ssh/sshd_config \
 && sed -i "s/#\(Port \).*/\1$port/g" /etc/ssh/sshd_config


# Copy training script
WORKDIR /workspace
COPY tf_mnist_hvd.py /workspace/tf_mnist_hvd.py

# Create mpiuser and home .ssh directory
RUN useradd -m -u ${UID} -s /bin/bash ${USER} \
 && mkdir -p /home/${USER}/.ssh \
 && chown -R ${UID}:${UID} /workspace /home/${USER}

# Switch to the non-root user
USER ${UID}

# Config for running sshd as non-root (custom config expects /etc/ssh host keys)
COPY --chown=${USER} sshd_config /home/${USER}/.sshd_config
RUN echo "Port $port" >> /home/mpiuser/.sshd_config

# Start sshd in foreground (required for MPI operator); tini for signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/usr/sbin/sshd", "-D", "-e", "-f","/home/${USER}/.sshd_config"]
