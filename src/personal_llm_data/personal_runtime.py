"""Small developer-runner guards, independent of the frozen studies."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess


@contextmanager
def gpu_guard(root):
    """OS lock is released on exit/crash. Also reject other NVIDIA compute PIDs."""
    path = Path(root)/"experiments/personal_adapter_v1/gpu.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        locked = False
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
            inventory = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                         "--format=csv,noheader,nounits"], text=True, timeout=15)
            lines = [line.strip() for line in inventory.splitlines() if line.strip()]
            if any(not line.isdigit() for line in lines):
                raise RuntimeError("Cannot establish NVIDIA compute-process inventory")
            if any(int(line) != os.getpid() for line in lines):
                raise RuntimeError("Another GPU compute process exists; inspect before executing")
            yield
        finally:
            if locked:
                stream.seek(0)
                if os.name == "nt":
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def checkpoint_inventory(path):
    from personal_llm_data.personal_data import sha
    path = Path(path)
    required = {"adapter_model.safetensors", "adapter_config.json", "optimizer.pt", "scheduler.pt", "trainer_state.json", "rng_state.pth"}
    if not required.issubset({p.name for p in path.iterdir() if p.is_file()}):
        raise ValueError("Checkpoint missing adapter/optimizer/scheduler/RNG/trainer state")
    state = json.loads((path/"trainer_state.json").read_text(encoding="utf-8"))
    if type(state.get("global_step")) is not int or state["global_step"] < 1:
        raise ValueError("Checkpoint must record a positive optimizer step")
    return dict(step=state["global_step"], files_sha256={name: sha(path/name) for name in sorted(required)})
