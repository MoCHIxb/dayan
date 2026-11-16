#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0
[[ -x .venv-cloud8b/bin/python ]] || { echo 'Prepare dedicated8B environment first' >&2; exit 2; }
exec .venv-cloud8b/bin/python -u scripts/run_qwen8b_banking500.py "$@"
