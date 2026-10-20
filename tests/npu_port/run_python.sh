#!/usr/bin/env bash
# Isolated source imports; no global pip install or LightGBM path changes.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
qlib_root=${QLIB_XGB_ROOT:-/workspace/qlib-xgb-npu}
python_bin=${XGB_NPU_PYTHON:-/workspace/xgb-npu-venv/bin/python}
export PYTHONPATH="$root/python-package:$qlib_root"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
exec "$python_bin" "$@"