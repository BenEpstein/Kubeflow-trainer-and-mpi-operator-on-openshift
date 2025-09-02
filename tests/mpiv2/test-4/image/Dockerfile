# Simplified Dockerfile without distro Open MPI - use HPC-X from the base image
FROM nvcr.io/nvidia/pytorch:24.05-py3

ARG USER=mpiuser
ARG UID=10001
ARG port=2222

# Network-hardy apt
RUN echo 'Acquire::ForceIPv4 "true";' > /etc/apt/apt.conf.d/99force-ipv4 \
 && echo 'Acquire::Retries "6";'      > /etc/apt/apt.conf.d/80retries \
 && sed -i 's|http://|https://|g' /etc/apt/sources.list

# Install SSH, build tools, tini (no openmpi here!)
RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-server openssh-client \
        build-essential python3-dev \
        ca-certificates curl vim tini \
    && rm -rf /var/lib/apt/lists/*

# Use HPC-X Open MPI bundled in the NGC image
ENV HPCX=/opt/hpcx
ENV PATH=$HPCX/ompi/bin:$PATH
ENV LD_LIBRARY_PATH=$HPCX/ompi/lib:$HPCX/ucx/lib:$LD_LIBRARY_PATH

# Python deps
RUN pip install --no-cache-dir \
    torchvision \
    mpi4py \
 && pip install --no-cache-dir \
    'protobuf>=3.20.0,<3.21' \
    'onnx>=1.14,<1.16'

# SSH daemon setup
RUN mkdir -p /var/run/sshd /etc/ssh \
 && sed -i "s/[ #]\(.*StrictHostKeyChecking \).*/\1no/g" /etc/ssh/ssh_config \
 && echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config \
 && sed -i "s/[ #]\(.*Port \).*/\1$port/g" /etc/ssh/ssh_config \
 && sed -i "s/#\(StrictModes \).*/\1no/g" /etc/ssh/sshd_config \
 && sed -i "s/#\(Port \).*/\1$port/g" /etc/ssh/sshd_config

WORKDIR /workspace
COPY torch_mnist_ddp.py /workspace/torch_mnist_ddp.py

RUN useradd -m -u ${UID} -s /bin/bash ${USER} \
 && mkdir -p /home/${USER}/.ssh \
 && chown -R ${UID}:${UID} /workspace /home/${USER}

USER ${UID}
COPY --chown=${USER} sshd_config /home/${USER}/.sshd_config
RUN echo "Port $port" >> /home/${USER}/.sshd_config

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/usr/sbin/sshd", "-D", "-e", "-f", "/home/${USER}/.sshd_config"]
