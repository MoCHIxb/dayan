#!/usr/bin/env bash
# Run from the checkout. Installing dependencies is a separate explicit step.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0
action="${1:-plan}"
if [[ $# -gt 0 ]]; then shift; fi
case "$action" in
  bootstrap)
    [[ "$(uname -s)" == "Linux" ]] || { echo 'Bootstrap requires the designated Linux server' >&2; exit 2; }
    nvidia-smi --query-gpu=name --format=csv,noheader
    [[ ! -e .venv ]] || { echo 'Existing .venv retained; activate and inspect it, do not overwrite' >&2; exit 2; }
    python3.12 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements-train.lock
    python -m pip install -e . --no-deps
    python -m pip check
    python -m unittest discover -s tests -q
    ;;
  plan) python scripts/run_intenttune_3090.py "$@" ;;
  prepare) python scripts/run_intenttune_3090.py --prepare "$@" ;;
  check) python scripts/run_intenttune_3090.py --check "$@" ;;
  run) python scripts/run_intenttune_3090.py --execute "$@" ;;
  resume) python scripts/run_intenttune_3090.py --execute --resume "$@" ;;
  report) python scripts/run_intenttune_3090.py --report "$@" ;;
  pause) python scripts/run_intenttune_3090.py --pause "$@" ;;
  *) echo 'Usage: bash scripts/cloud/run_banking77_3090.sh {bootstrap|plan|prepare|check|run|resume|report|pause} [options]' >&2; exit 2 ;;
esac
