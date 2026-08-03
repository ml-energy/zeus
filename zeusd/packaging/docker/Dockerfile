# Zeus daemon (zeusd) container image.
#
# Build from the zeusd/ directory:
#   docker build -f packaging/docker/Dockerfile -t mlenergy/zeusd .
#
# NVML is injected at runtime by the NVIDIA Container Toolkit, and the AMD SMI
# library is bundled on amd64 (no ROCm packages exist for arm64).
# See https://ml.energy/zeus/zeusd/ for run recipes.

ARG TARGETARCH

FROM rust:1-slim AS builder
WORKDIR /zeusd
COPY . .
RUN cargo build --release --locked && strip target/release/zeusd

# The amd-smi-lib deb drags in Python for the amd-smi CLI, but zeusd only
# dlopens libamd_smi.so, so extract just the library from the package.
FROM ubuntu:24.04 AS amdlib-amd64
ARG ROCM_APT_VERSION=7.2
RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends ca-certificates wget gpg \
    && mkdir -p /etc/apt/keyrings \
    && wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key \
       | gpg --dearmor > /etc/apt/keyrings/rocm.gpg \
    && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_APT_VERSION} noble main" \
       > /etc/apt/sources.list.d/rocm.list \
    && apt-get update -qq \
    && apt-get download amd-smi-lib \
    && dpkg-deb -x amd-smi-lib_*.deb /amd-smi-lib \
    && mkdir /amdsmi \
    && cp -a /amd-smi-lib/opt/rocm-*/lib/libamd_smi.so.* /amdsmi/

FROM ubuntu:24.04 AS amdlib-arm64
RUN mkdir /amdsmi

FROM amdlib-${TARGETARCH} AS amdlib

FROM ubuntu:24.04
COPY --from=amdlib /amdsmi/ /usr/local/lib/amdsmi/
COPY --from=builder /zeusd/target/release/zeusd /usr/local/bin/zeusd
RUN mkdir -p /run/zeusd
ENV AMDSMI_LIB_DIR=/usr/local/lib/amdsmi
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=utility
ENTRYPOINT ["zeusd"]
CMD ["serve"]
