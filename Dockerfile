# syntax=docker/dockerfile:1
# Multi-stage build for mamba-onnx wheels (linux-x86_64 + macos-x86_64)

ARG ZIG_VERSION=0.15.1
ARG PYTHON_VERSION=3.12

# ---------------------------------------------------------------------------
# Stage 1: System deps + toolchain (cached independently of source)
# ---------------------------------------------------------------------------
FROM ubuntu:24.04 AS toolchain

ARG ZIG_VERSION
ARG PYTHON_VERSION

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl xz-utils \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -fsSL https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Install zvm + zig
ENV ZVM_INSTALL="/root/.zvm/self"
ENV PATH="/root/.zvm/bin:${ZVM_INSTALL}:${PATH}"
RUN curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/tristanisham/zvm/master/install.sh | bash \
    && zvm install ${ZIG_VERSION}

# ---------------------------------------------------------------------------
# Stage 2: Python venv + source (rebuilds when source changes)
# ---------------------------------------------------------------------------
FROM toolchain AS builder

ARG PYTHON_VERSION

WORKDIR /src
COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --python python${PYTHON_VERSION} /venv \
    && . /venv/bin/activate \
    && uv pip install "setuptools>=61" build wheel

ENV VIRTUAL_ENV=/venv
ENV PATH="/venv/bin:${PATH}"

# ---------------------------------------------------------------------------
# Stage 3: Build Linux x86_64 wheel
# ---------------------------------------------------------------------------
FROM builder AS build-linux

RUN --mount=type=cache,target=/root/.cache/zig \
    --mount=type=cache,target=/root/.cache/uv \
    . /venv/bin/activate \
    && python -m build --wheel --no-isolation \
       -C optimize=ReleaseFast \
       -C plat-name=manylinux_2_17_x86_64 \
       -o /wheels

# ---------------------------------------------------------------------------
# Stage 4: Build macOS x86_64 wheel (cross-compiled via Zig)
# ---------------------------------------------------------------------------
FROM builder AS build-macos

RUN --mount=type=cache,target=/root/.cache/zig \
    --mount=type=cache,target=/root/.cache/uv \
    . /venv/bin/activate \
    && python -m build --wheel --no-isolation \
       -C optimize=ReleaseFast \
       -C target=x86_64-macos \
       -C cpu=x86_64_v3 \
       -C plat-name=macosx-11.0-x86_64 \
       -o /wheels

# ---------------------------------------------------------------------------
# Stage 5: Collect all wheels into a scratch image
# ---------------------------------------------------------------------------
FROM scratch AS artifacts

COPY --from=build-linux /wheels/*.whl /
COPY --from=build-macos /wheels/*.whl /
