"""One local entry point for the source-checkout adaptation workflow."""
from __future__ import annotations
import os
from pathlib import Path
import subprocess
import sys

COMMANDS = {
    'prepare': 'prepare_lamp7.py',
    'train-personal': 'train_personal_adapter.py',
    'evaluate-personal': 'evaluate_personal_adapter.py',
    'train': 'train_qwen_lora.py',
    'evaluate': 'evaluate_generation.py',
    'infer': 'infer_qwen.py',
    'score-classification': 'score_classification.py',
    'audit-peft': 'run_peft_sequence.py',
}


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ('-h','--help'):
        print('personal-llm <plan|prepare-personal|train-personal|evaluate-personal|score-classification|prepare|train|evaluate|infer|audit-peft> [arguments]\n'
              'Use <command> --help for its options. Requires the source checkout;\n'
              'audit-peft is read-only unless explicitly passed --freeze/--run/--resume.')
        return 0
    if args[0] == 'prepare-personal':
        from personal_llm_data.personal_data import main as prepare_main
        return prepare_main(args[1:])
    if args[0] == 'plan':
        from personal_llm_data.planning import main as plan_main
        return plan_main(args[1:])
    if args[0] not in COMMANDS:
        raise SystemExit(f'Unknown command: {args[0]}')
    root = Path(__file__).resolve().parents[2]
    script = root/'scripts'/COMMANDS[args[0]]
    if not script.is_file():
        raise SystemExit('Workflow scripts unavailable: use an editable source checkout, not a standalone wheel')
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(root/'src') + (os.pathsep+environment['PYTHONPATH'] if environment.get('PYTHONPATH') else '')
    result = subprocess.run([sys.executable,'-u',str(script),*args[1:]],cwd=root,env=environment)
    return result.returncode


if __name__=='__main__':
    raise SystemExit(main())
