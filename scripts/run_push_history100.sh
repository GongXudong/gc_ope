#!/usr/bin/env bash
# 四方法串行；每方法五个 seed 并行；每 seed 内四个 checkpoint 并行。
set -euo pipefail
task_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec conda run --no-capture-output -n gc_ope env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u "${task_script_dir}/run_push_history100.py" "$@"
