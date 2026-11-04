# syntax=docker/dockerfile:1

# V100: --build-arg CUDA_VERSION=12.8.1 --build-arg NINFER_TARGET=v100
# Set BUILD_JOBS to fit the builder's memory (for example, 3 on a 32 GiB host).
ARG CUDA_VERSION=13.1.2
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS build

ARG DEBIAN_FRONTEND=noninteractive
ARG NINFER_TARGET=rtx5090
ARG BUILD_JOBS
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        cmake \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libcurl4-openssl-dev \
        libhwloc-dev \
        libjpeg-dev \
        libswscale-dev \
        liburing-dev \
        ninja-build \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Link against the toolkit stub when the build stage has no host GPU driver.
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so.1

WORKDIR /src
COPY . .

RUN cmake -S . -B /build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DNINFER_TARGET=${NINFER_TARGET} \
        -DNINFER_BUILD_APPS=ON \
        -DBUILD_TESTING=OFF \
        -DNINFER_BUILD_BENCHMARKS=OFF \
    && cmake --build /build --parallel ${BUILD_JOBS} --target ninfer ninfer-serve

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        libavcodec60 \
        libavformat60 \
        libavutil58 \
        libcurl4t64 \
        libhwloc15 \
        libjpeg-turbo8 \
        libswscale7 \
        liburing2 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /build/apps/ninfer /usr/local/bin/ninfer
COPY --from=build /build/apps/ninfer-serve /usr/local/bin/ninfer-serve
COPY --from=build /build/lib/ /usr/local/lib/

WORKDIR /workspace
EXPOSE 8080
STOPSIGNAL SIGTERM

CMD ["ninfer-serve", "--help"]
