#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mode=${1:?Usage: run_time.sh server|eval|test|check-context GPU [arguments]}
gpu=${2:?Specify GPU index}
shift 2
if [[ -n "${TIME_PYTHON:-}" ]]; then
    python_bin=$TIME_PYTHON
elif [[ -x "$repo_dir/.venv/bin/python" ]]; then
    python_bin="$repo_dir/.venv/bin/python"
else
    python_bin=python3
fi
export PYTHONPATH="$repo_dir/src:$repo_dir/packages/openpi-client/src:${LIBERO_SOURCE:-$repo_dir/third_party/libero}${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/openpi}
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export CUDA_VISIBLE_DEVICES="$gpu"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$repo_dir"
case "$mode" in
    server) export JAX_PLATFORMS=cuda; script=scripts/serve_time.py ;;
    check-context) export JAX_PLATFORMS=${TIME_JAX_PLATFORMS:-cuda}; script=scripts/check_time_context.py ;;
    eval)
        export JAX_PLATFORMS=cpu MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
        export MUJOCO_EGL_DEVICE_ID="$gpu"
        export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-$HOME/.libero}
        export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
        script=examples/libero/eval_time.py ;;
    test) export JAX_PLATFORMS=cpu; exec "$python_bin" -m unittest discover -s tests -p 'test_time_*.py' -v ;;
    *) printf 'Unknown mode: %s\n' "$mode"; exit 2 ;;
esac
exec "$python_bin" "$script" "$@"
