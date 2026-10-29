#!/usr/bin/env bash
# Builds the title folder in the pinned Debian x86-64 container that CI uses
# (docs/CI.md), on any Docker host: on an Apple Silicon Mac, Colima or Docker
# Desktop with Rosetta for amd64 images (colima start --vm-type vz --vz-rosetta).
#
#   scripts/build-in-container.sh      build/app-ps5/title/PPSA99575
#   BUILD_JOBS=4 ...                   compiler jobs (default: the Docker VM's CPUs)
#   PS5LM_PREBUILT_UI=OFF ...          omit llama-server's browser chat UI
#
# First runs scripts/test-native-build.sh (pinned SDK, runtime, patched RADV,
# the title and its static checks, without the browser UI), then packages again
# with the browser UI as a release has it. Dependencies stay in the Docker
# volume ps5lm-deps between runs; `docker volume rm ps5lm-deps` starts over.
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
image=ps5lm-build:trixie
jobs=${BUILD_JOBS:-$(docker info --format '{{.NCPU}}')}

# The package list of .github/workflows/native-build.yml, and patch for patches/sdk.
docker build --platform linux/amd64 -t "$image" - <<'EOF'
FROM debian:trixie-20261020-slim@sha256:62bd423aeee912639ca8ee681c2b762cb3e25137e0a92b20be455898a9f034ad
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git ca-certificates wget curl unzip rsync cmake ninja-build \
      clang-19 llvm-19-dev lld-19 libclang-rt-19-dev libclang-19-dev libclang-cpp19-dev \
      libllvmspirvlib-19-dev libclc-19-dev spirv-tools spirv-headers glslang-tools \
      meson pkgconf flex bison python3 python3-mako python3-yaml python3-packaging python3-setuptools \
      zlib1g-dev libzstd-dev libffi-dev libedit-dev libxml2-dev libvulkan-dev glslc ffmpeg xxd patch \
    && rm -rf /var/lib/apt/lists/*
ENV PATH=/usr/lib/llvm-19/bin:$PATH LLVM_CONFIG=/usr/bin/llvm-config-19
EOF

# The checkout at its own path, and the repository's git directory too: a
# worktree's .git and its submodules point there by absolute path.
git_dir=$(cd "$root" && cd "$(git rev-parse --git-common-dir)" && pwd)
docker run --rm --platform linux/amd64 -v "$root:$root" -v "$git_dir:$git_dir" -v ps5lm-deps:/deps -w "$root" \
    -e BUILD_JOBS="$jobs" -e PS5LM_NATIVE_SOURCES=/deps -e PS5LM_PREBUILT_UI="${PS5LM_PREBUILT_UI:-ON}" \
    -e PS5LM_STEP_TIMEOUT="${PS5LM_STEP_TIMEOUT:-14400}" \
    "$image" bash -euo pipefail -c '
        # The checkout belongs to the host user, not the container root.
        for d in "$PWD" "$PWD/third_party/llama.cpp" "$PWD/third_party/ps5-homebrew-ui"; do
            git config --global --add safe.directory "$d"
        done
        prebuilt_ui=$PS5LM_PREBUILT_UI
        scripts/test-native-build.sh
        if [ "$prebuilt_ui" != OFF ]; then
            PS5_VULKAN_ROOT=/deps/PS5_Vulkan PS5_PAYLOAD_SDK_FORK=/deps/PS5_PayloadSDK \
                CMAKE_BUILD_PARALLEL_LEVEL="$BUILD_JOBS" PS5LM_PREBUILT_UI=ON scripts/build-app.sh
        fi
    '
title_id=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["titleId"])' "$root/ps5/app/sce_sys/param.json")
echo "build-in-container: $root/build/app-ps5/title/$title_id"
