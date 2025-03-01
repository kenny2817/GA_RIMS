FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

# Minimal dependencies for Singularity
RUN apt-get update && apt-get install -y \
    build-essential \
    libseccomp-dev \
    squashfs-tools \
    wget \
    git \
    curl \
    libgpgme-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Go
RUN curl -fsSL https://go.dev/dl/go1.21.5.linux-amd64.tar.gz | tar -xz -C /usr/local
ENV PATH="/usr/local/go/bin:${PATH}"

# Install Singularity
WORKDIR /opt
RUN export VERSION=3.8.7 && \
    wget https://github.com/hpcng/singularity/releases/download/v${VERSION}/singularity-${VERSION}.tar.gz && \
    tar -xzf singularity-${VERSION}.tar.gz

WORKDIR /opt/singularity-3.8.7
RUN ./mconfig && \
    make -C builddir && \
    make -C builddir install

# Default shell
CMD ["/bin/bash"]
