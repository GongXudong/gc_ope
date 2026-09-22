#!/usr/bin/env bash
# 从任何目录都可启动；实际计算固定使用 gc_ope Conda 环境。
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec conda run --no-capture-output -n gc_ope python "$script_dir/run_push_all100.py" "$@"
