#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
[[ -x .venv-cloud8b/bin/python ]] || { echo 'Prepare .venv-cloud8b first' >&2; exit 2; }
# HF retains academic proxy for small fixed metadata; ModelScope/CDN bypass it.
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}modelscope.cn,.modelscope.cn,modelscope.net,.modelscope.net,aliyuncs.com,.aliyuncs.com"
export no_proxy="$NO_PROXY"
exec .venv-cloud8b/bin/python -u scripts/deploy_research_models.py "$@"
