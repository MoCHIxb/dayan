#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_DISABLE_XET=1
export PYTHONDONTWRITEBYTECODE=1
action="${1:-plan}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ "$action" == bootstrap ]]; then
  [[ "$(uname -s)" == Linux ]] || exit 2
  [[ ! -e .venv-cloud8b ]] || { echo 'Existing .venv-cloud8b retained' >&2; exit 2; }
  python3 -c 'import sys,torch; assert sys.version_info[:2]==(3,12); assert torch.__version__.startswith("2.12.1"); assert torch.version.cuda=="13.0"'
  python3 -m venv --system-site-packages .venv-cloud8b
  .venv-cloud8b/bin/python -c 'import torch,pathlib; p=pathlib.Path(".venv-cloud8b/image-torch.constraints.txt"); p.write_text("torch==="+torch.__version__+"\n")'
  .venv-cloud8b/bin/python -m pip install -r requirements-cloud8b.lock -c .venv-cloud8b/image-torch.constraints.txt
  .venv-cloud8b/bin/python -m pip install -e . --no-deps
  .venv-cloud8b/bin/python -m pip check
elif [[ "$action" == plan ]]; then
  python3 scripts/prepare_8b_4090.py plan "$@"
elif [[ "$action" == check || "$action" == pin || "$action" == download || "$action" == verify ]]; then
  [[ -x .venv-cloud8b/bin/python ]] || { echo 'Run bootstrap first' >&2; exit 2; }
  .venv-cloud8b/bin/python scripts/prepare_8b_4090.py "$action" "$@"
else
  echo 'Only preparation: bootstrap|plan|check|pin|download|verify; training runner pending' >&2
  exit 2
fi
