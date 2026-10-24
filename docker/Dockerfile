# Muse Spark recipe image (muse-tf): the pinned TensorFold GB10 CUDA runtime (same base as the MiMo and GLM recipes)
# plus exllamav3 at a pinned commit with this recipe's patches (patches-exl3/: the muse_spark text architecture).
# exllamav3 is the quantiser for our EXL3 packs and the reference forward the Zig engine is gated against.
#
# Build with scripts/build-image.sh (tags muse-tf:<patched exllamav3 sha> and muse-tf:latest). Never pull the base:
# it is present locally on both nodes by digest.
ARG BASE_IMAGE=ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold@sha256:14f15591eae5d6a540f09218d3852068962fe5381371bbfefe0e9194cd834529
FROM ${BASE_IMAGE}

# 1. exllamav3 at the pinned commit, CUDA extension compiled for GB10 (sm_121). --no-deps: the base's PyTorch is an
#    NVIDIA prerelease build that pip must never replace. Kept apart from the patches so a patch change does not
#    recompile the ~175 CUDA sources (the patches are Python-only; step 2 checks that).
ARG EXL3_REPO=https://github.com/turboderp-org/exllamav3
ARG EXL3_COMMIT=058b8630fd081f3ff46fce6246955b157c3a2e3b
ARG MAX_JOBS=12
RUN git clone -q "${EXL3_REPO}" /opt/exllamav3 && git -C /opt/exllamav3 checkout -q "${EXL3_COMMIT}" \
 && cd /opt/exllamav3 && TORCH_CUDA_ARCH_LIST="12.1" MAX_JOBS=${MAX_JOBS} \
    pip install --no-deps --no-build-isolation --no-cache-dir . \
 && pip install --no-deps --no-cache-dir marisa_trie llguidance \
 && python3 -c "import exllamav3, exllamav3_ext; from importlib.metadata import version; print('exllamav3', version('exllamav3'))"

# 2. The recipe's patches, applied as commits (Python runs from / so it imports the installed package, not the tree) (same identity and dates as patches-exl3/ was exported with, so HEAD
#    matches the patched sha the image is tagged with), then copied over the installed Python package.
COPY patches-exl3 /opt/patches-exl3
RUN cd /opt/exllamav3 \
 && git -c user.name="ajensenwaud" -c user.email="ajensenwaud@users.noreply.github.com" \
        am -q --committer-date-is-author-date /opt/patches-exl3/*.patch \
 && if git diff --name-only "${EXL3_COMMIT}" HEAD | grep -qv '\.py$'; then \
        echo "patches touch non-Python files: rebuild the extension"; \
        TORCH_CUDA_ARCH_LIST="12.1" MAX_JOBS=${MAX_JOBS} pip install --no-deps --no-build-isolation --no-cache-dir . ; \
    else \
        site=$(cd / && python3 -c "import exllamav3, os; print(os.path.dirname(os.path.dirname(exllamav3.__file__)))") \
        && git diff --name-only --diff-filter=AM "${EXL3_COMMIT}" HEAD | while read -r f; do \
               mkdir -p "$site/$(dirname "$f")" && cp "$f" "$site/$f" && echo "patched $f"; done ; \
    fi \
 && git rev-parse HEAD > /opt/exllamav3.patched-sha && git rev-parse 'HEAD^{tree}' > /opt/exllamav3.patched-tree \
 && cd / && python3 -c "from exllamav3.architecture.architectures import get_architectures as g; \
assert 'MuseSparkForConditionalGeneration' in g(); print('muse_spark registered, exllamav3', open('/opt/exllamav3.patched-sha').read().strip())"

# 3. Converter wrapper (clears sys.argv after parsing; see tools/exl3/convert.py)
COPY tools/exl3 /tools
ARG EXL3_PATCHED_SHA=unknown
ARG RECIPE_COMMIT=unknown
LABEL org.opencontainers.image.title="muse-tf" org.opencontainers.image.revision="${RECIPE_COMMIT}" \
      io.exllamav3.revision="${EXL3_COMMIT}" io.exllamav3.patched="${EXL3_PATCHED_SHA}"
WORKDIR /workspace
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh", "python3"]
