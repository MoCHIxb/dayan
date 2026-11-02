"""Small developer-runner guards, independent of the frozen studies."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import ntpath
import xml.etree.ElementTree as ET
import re


# WDDM 610.x can expose desktop renderers through --query-compute-apps and
# classify all of them C+G. Only known desktop names are excepted; Python,
# model services, C-only entries and unknown live processes remain blocked.
WDDM_DESKTOP_NAMES = frozenset(name.casefold() for name in (
    'dwm.exe', 'explorer.exe', 'SystemSettings.exe', 'LockApp.exe',
    'CrossDeviceResume.exe', 'ShellExperienceHost.exe', 'StartMenuExperienceHost.exe',
    'TextInputHost.exe', 'SearchHost.exe', 'ShellHost.exe', 'ApplicationFrameHost.exe', 'TabTip.exe',
    'NVIDIA Overlay.exe', 'msedgewebview2.exe', 'Code.exe', 'ChatGPT.exe',
    'codex-computer-use-swift.exe', 'webwallpaper64.exe', 'AMDRSSrcExt.exe', 'RadeonSoftware.exe'))


def wddm_desktop_exclusions(pids, inventory_xml, lookup=None):
    """Return verified desktop/stale PID exceptions, never infer from C+G alone.

    lookup(pid) returns a live OS process name, None only if absent, and raises
    when access is denied. XML names alone cannot hide a reused/unknown PID.
    """
    try:
        root = ET.fromstring(inventory_xml)
    except ET.ParseError:
        return {}
    devices = root.findall('gpu')
    if not devices or any(g.findtext('driver_model/current_dm') != 'WDDM' for g in devices):
        return {}
    if lookup is None:
        import psutil
        def lookup(pid):
            try:
                process = psutil.Process(pid)
                try:
                    return process.exe()
                except psutil.AccessDenied:
                    return process.name()
            except psutil.NoSuchProcess:
                return None
    records = {int(r.findtext('pid')): r for g in devices for r in g.findall('processes/process_info')
               if (r.findtext('pid') or '').isdigit()}
    allowed = {}
    for pid in pids:
        try:
            name = lookup(pid)
        except (PermissionError, OSError):
            continue
        # psutil.AccessDenied may not inherit OSError; preserve that failure.
        if name is None:
            allowed[pid] = 'stale NVIDIA entry; OS process absent'
            continue
        row = records.get(pid)
        actual = ntpath.basename(name).casefold()
        # These two installed GIGABYTE utilities were checked against local
        # Authenticode/Appx publisher metadata. Match protected vendor paths,
        # never allow an arbitrary program named GCC.exe/GBT_DL_LIB.exe.
        vendor_path = bool(re.fullmatch(
            r'[a-z]:\\program files\\(?:gigabyte\\control center\\gcc\.exe|windowsapps\\65d483df-b37e-4fcf-94de-8b795233db63_[^\\]+__1mmjbktjj1mkp\\gbt_dl_lib\.exe)',
            ntpath.normpath(name).casefold()))
        if row is None or row.findtext('type') not in ('G', 'C+G') or (actual not in WDDM_DESKTOP_NAMES and not vendor_path):
            continue
        recorded = ntpath.basename(row.findtext('process_name') or '').casefold()
        if recorded and recorded != actual:
            continue
        allowed[pid] = 'WDDM desktop graphics: ' + name
    return allowed


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
            other_pids = {int(line) for line in lines if int(line) != os.getpid()}
            if other_pids and os.name == 'nt':
                inventory_xml = subprocess.check_output(['nvidia-smi', '-q', '-x'], text=True, timeout=15)
                exceptions = wddm_desktop_exclusions(other_pids, inventory_xml)
                other_pids -= set(exceptions)
                if exceptions:
                    print('WDDM preflight: verified desktop/stale entries excluded: ' +
                          ', '.join(f'{pid}:{name}' for pid, name in sorted(exceptions.items())), flush=True)
            if other_pids:
                raise RuntimeError("Another GPU compute/unknown process exists; inspect before executing")
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
