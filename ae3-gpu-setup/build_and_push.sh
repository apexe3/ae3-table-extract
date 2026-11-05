#!/bin/bash
# AE3-TL — Open-Source Table Extraction Toolkit
# Copyright 2026 Team APEX:E3
# Licensed under the Apache License, Version 2.0; see LICENSE in the repository root.
#
# Builds the AE3-TL images and pushes them to Docker Hub.
#   apexe3/ae3-table-extract:gpu  and :0.23-gpu   (x86_64, NVIDIA CUDA)
#   apexe3/ae3-table-extract:cpu  and :0.23-cpu   (x86_64 and ARM, e.g. Apple silicon)
#
# Run `docker login` first, then:   ./build_and_push.sh
# Run it on an x86_64 Linux server: on an Apple-silicon Mac the GPU image builds under emulation,
# which downloads and unpacks about 2 GB of NVIDIA libraries very slowly.
# Override with:  VERSION=0.24 REPO=apexe3/ae3-table-extract CPU_PLATFORMS=linux/amd64 ./build_and_push.sh
set -euo pipefail

VERSION="${VERSION:-0.23}"
REPO="${REPO:-apexe3/ae3-table-extract}"
CPU_PLATFORMS="${CPU_PLATFORMS:-linux/amd64,linux/arm64}"

cd "$(dirname "$0")"

# A buildx builder that can build for several platforms and push
if ! docker buildx inspect ae3-builder >/dev/null 2>&1; then
  docker buildx create --name ae3-builder --driver docker-container >/dev/null
fi
docker buildx use ae3-builder

# On an x86 Linux server, building the ARM half of the CPU image needs QEMU (Docker Desktop has it)
if [[ "$CPU_PLATFORMS" == *arm64* && "$(uname -s)" == "Linux" && "$(uname -m)" != "aarch64" ]]; then
  docker run --privileged --rm tonistiigi/binfmt --install arm64 >/dev/null
fi

echo "== GPU image: $REPO:gpu, $REPO:$VERSION-gpu"
docker buildx build --platform linux/amd64 -f Dockerfile \
  --label org.opencontainers.image.version="$VERSION" \
  -t "$REPO:gpu" -t "$REPO:$VERSION-gpu" --push .

echo "== CPU image: $REPO:cpu, $REPO:$VERSION-cpu ($CPU_PLATFORMS)"
docker buildx build --platform "$CPU_PLATFORMS" -f Dockerfile.cpu \
  --label org.opencontainers.image.version="$VERSION" \
  -t "$REPO:cpu" -t "$REPO:$VERSION-cpu" --push .

echo "Done:"
echo "  docker pull $REPO:gpu    # or $REPO:$VERSION-gpu"
echo "  docker pull $REPO:cpu    # or $REPO:$VERSION-cpu"
