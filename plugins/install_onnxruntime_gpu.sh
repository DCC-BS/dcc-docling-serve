#!/bin/sh
# Make RapidOCR (onnxruntime backend) run on the GPU in the CUDA images.
#
#   install_onnxruntime_gpu.sh install        (as the image user)
#     Swap the CPU-only onnxruntime of the upstream image for the onnxruntime-gpu
#     build matching the CUDA version torch ships with.
#   install_onnxruntime_gpu.sh register-libs  (as root)
#     Register the CUDA/cuDNN libraries of the nvidia-* wheels with the dynamic
#     linker. Without this the CUDA provider only loads when torch happened to be
#     imported first, and onnxruntime otherwise falls back to the CPU silently.
#
# Both steps are no-ops on CPU images (torch without CUDA).
set -eu

CUDA_VERSION="$(python -c 'import torch; print(torch.version.cuda or "")')"
if [ -z "$CUDA_VERSION" ]; then
    echo "torch has no CUDA support: keeping the CPU onnxruntime"
    exit 0
fi

case "${1:-}" in
install)
    ORT_VERSION="$(python -c 'import importlib.metadata as m; print(m.version("onnxruntime"))')"
    echo "torch CUDA ${CUDA_VERSION}, replacing onnxruntime ${ORT_VERSION} with onnxruntime-gpu"
    pip uninstall -y onnxruntime

    # The CUDA runtime, cuBLAS and cuDNN come from the nvidia-* wheels torch already
    # installed, so onnxruntime-gpu goes in without its optional CUDA extras (--no-deps).
    case "$CUDA_VERSION" in
    13.*)
        # PyPI builds of onnxruntime-gpu target CUDA 13; stay on the upstream version if possible.
        pip install --no-cache-dir --no-deps "onnxruntime-gpu==${ORT_VERSION}" \
            || pip install --no-cache-dir --no-deps onnxruntime-gpu
        ;;
    12.*)
        pip install --no-cache-dir --no-deps --index-url "${ONNXRUNTIME_CUDA12_INDEX}" onnxruntime-gpu
        ;;
    *)
        echo "No onnxruntime-gpu build known for CUDA ${CUDA_VERSION}" >&2
        exit 1
        ;;
    esac

    python -c '
import onnxruntime
providers = onnxruntime.get_available_providers()
print("onnxruntime", onnxruntime.__version__, providers)
assert "CUDAExecutionProvider" in providers, "onnxruntime-gpu has no CUDAExecutionProvider"
'
    ;;
register-libs)
    SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    find "${SITE_PACKAGES}/nvidia" -mindepth 2 -maxdepth 3 -type d -name lib | sort \
        > /etc/ld.so.conf.d/zz-nvidia-python-wheels.conf
    cat /etc/ld.so.conf.d/zz-nvidia-python-wheels.conf
    ldconfig

    # libcuda.so.1 is the driver library, mounted by the NVIDIA runtime when the container starts.
    PROVIDER="$(python -c 'import onnxruntime, pathlib; print(pathlib.Path(onnxruntime.__file__).parent / "capi" / "libonnxruntime_providers_cuda.so")')"
    MISSING="$(ldd "${PROVIDER}" | grep 'not found' | grep -v 'libcuda\.so' || true)"
    if [ -n "${MISSING}" ]; then
        echo "onnxruntime CUDA provider has unresolved libraries:" >&2
        echo "${MISSING}" >&2
        exit 1
    fi
    echo "onnxruntime CUDA provider libraries resolve"
    ;;
*)
    echo "usage: $0 install|register-libs" >&2
    exit 2
    ;;
esac
