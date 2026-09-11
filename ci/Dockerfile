# SPDX-License-Identifier: AGPL-3.0-only
FROM ubuntu:24.04@sha256:224a1869083a311ef3f13648a154ba79832fbef6364d31493642ca03082da254

ARG QEMU_VERSION=11.0.3
ARG QEMU_SHA256=da5fcffc32762820568b828ed430a728864d34d50b6d2f30358597760cbb0523
ARG TORCH_VERSION=2.7.1

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
        build-essential ca-certificates cmake curl gcc-aarch64-linux-gnu \
        libc6-dev-arm64-cross libglib2.0-dev libjson-c-dev libpixman-1-dev \
        ninja-build pkg-config python3 python3-dev python3-pip python3-venv \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/cpu2tensor-venv

WORKDIR /tmp/qemu
RUN curl --fail --location --silent --show-error \
        "https://download.qemu.org/qemu-${QEMU_VERSION}.tar.xz" \
        --output qemu.tar.xz \
    && echo "${QEMU_SHA256}  qemu.tar.xz" | sha256sum --check --strict \
    && tar --extract --file qemu.tar.xz --strip-components=1 \
    && mkdir build \
    && cd build \
    && ../configure \
        --prefix=/opt/qemu \
        --target-list=x86_64-linux-user,aarch64-linux-user \
        --enable-plugins \
        --disable-docs \
        --disable-system \
        --disable-tools \
        --without-default-features \
    && ninja qemu-x86_64 qemu-aarch64 \
    && install -D -m 0755 qemu-x86_64 /opt/qemu/bin/qemu-x86_64 \
    && install -D -m 0755 qemu-aarch64 /opt/qemu/bin/qemu-aarch64 \
    && install -D -m 0644 ../include/plugins/qemu-plugin.h \
        /opt/qemu/include/qemu-plugin.h \
    && cd / \
    && rm -rf /tmp/qemu

ENV PATH="/opt/cpu2tensor-venv/bin:/opt/qemu/bin:${PATH}"
RUN python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==${TORCH_VERSION}" \
    && python -m pip install --no-cache-dir \
        coverage==7.10.6 gcovr==8.3 gymnasium==1.2.0 \
        pytest==8.4.2 pytest-cov==7.0.0 scikit-build-core==0.11.6

WORKDIR /workspace
