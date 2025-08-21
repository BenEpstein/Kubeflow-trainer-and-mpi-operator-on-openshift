# Dockerfile
# Pick an NGC TensorFlow image recent enough for CUDA 12.x. Example tag:
# nvcr.io/nvidia/tensorflow:24.08-tf2-py3  (check your registry access and tag)
FROM nvcr.io/nvidia/tensorflow:24.08-tf2-py3

# Create a non-root user
ARG USER=tfuser
ARG UID=10001
RUN useradd -m -u ${UID} -s /bin/bash ${USER}

# Install OpenSSH server and Open MPI + build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-server openssh-client \
        openmpi-bin libopenmpi-dev \
        python3-pip ca-certificates curl vim tini \
    && rm -rf /var/lib/apt/lists/*

# Horovod (uses MPI controller; NCCL on GPUs if present)
# The base image already has TF and CUDA libs.
RUN pip install --no-cache-dir horovod[tensorflow,mpi]

# SSHD setup (non-root home, keys will be mounted by MPIJob)
RUN mkdir -p /var/run/sshd /etc/ssh && \
    sed -ri 's/^#?PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config && \
    sed -ri 's/^#?PermitRootLogin .*/PermitRootLogin no/' /etc/ssh/sshd_config && \
    echo "AllowUsers ${USER}" >> /etc/ssh/sshd_config

# Allow sshd (a privileged port) as non-root via capability
# This keeps the container non-root but lets sshd bind to 22.
RUN apt-get update && apt-get install -y --no-install-recommends libcap2-bin && \
    setcap 'cap_net_bind_service=+ep' /usr/sbin/sshd && \
    rm -rf /var/lib/apt/lists/*

# Copy training script
WORKDIR /workspace
COPY tf_mnist_hvd.py /workspace/tf_mnist_hvd.py
RUN chown -R ${UID}:${UID} /workspace

# Switch to the non-root user
USER ${UID}

# Expose sshd
EXPOSE 22

# Start sshd in foreground (required for MPI operator); tini for signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/usr/sbin/sshd", "-D", "-e"]
