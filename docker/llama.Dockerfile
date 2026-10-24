# llama.cpp with Muse Spark Open support (ggml-org/llama.cpp PR #30616), CUDA for GB10 (sm_121a) plus the RPC
# backend, built on a local CUDA 13 base image (no registry pulls).
#
# Build context = a llama.cpp checkout at the pinned commit (see LLAMA_COMMIT), e.g.:
#   git -C /path/to/llama.cpp rev-parse HEAD        # must print LLAMA_COMMIT
#   docker build -f docker/llama.Dockerfile --build-arg JOBS=4 -t muse-llama:335916a /path/to/llama.cpp
#
# GB10 note: llama.cpp rewrites plain 12X archs to 12Xa (architecture-specific features such as the Blackwell
# MXFP4/FP4 MMA paths need the "a" variant); we pass 121a-real directly.
ARG BASE_IMAGE=tensorfold-glm53:v0.6.0
FROM ${BASE_IMAGE}

ARG LLAMA_COMMIT=335916ad3f0cab876a7a3cf16faaaaea28bf9a96
ARG CUDA_ARCH=121a-real
ARG JOBS=4

COPY . /opt/llama.cpp-src
WORKDIR /opt/llama.cpp-src

RUN cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=ON -DGGML_RPC=ON \
      -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
      -DGGML_NATIVE=ON \
      -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
 && cmake --build build -j ${JOBS} --target llama-server ggml-rpc-server llama-bench llama-cli llama-gguf-split \
 && mkdir -p /opt/llama/bin /opt/llama/lib \
 && cp build/bin/llama-server build/bin/ggml-rpc-server build/bin/llama-bench build/bin/llama-cli build/bin/llama-gguf-split /opt/llama/bin/ \
 && find build -name '*.so*' -exec cp -P {} /opt/llama/lib/ \; \
 && ln -s ggml-rpc-server /opt/llama/bin/rpc-server \
 && echo "${LLAMA_COMMIT}" > /opt/llama/COMMIT \
 && rm -rf build

ENV PATH=/opt/llama/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/llama/lib:${LD_LIBRARY_PATH}
WORKDIR /opt/llama
ENTRYPOINT []
