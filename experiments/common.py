import csv
import datetime as dt
import importlib.metadata
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = Path(os.environ.get("ALLM_RUN_ROOT", "~/allm-runs/allms-ass1")).expanduser()
CACHE = Path(os.environ.get("NANOCHAT_BASE_DIR", str(RUN_ROOT / "cache"))).expanduser()
LOGS = Path(os.environ.get("NANOCHAT_LOG_DIR", str(RUN_ROOT / "logs"))).expanduser()
CONSOLE = RUN_ROOT / "console"
RESULTS = RUN_ROOT / "results"
FIGURES = RUN_ROOT / "figures"
HARDWARE = RUN_ROOT / "hardware"
os.environ.setdefault("NANOCHAT_BASE_DIR", str(CACHE))
os.environ.setdefault("NANOCHAT_LOG_DIR", str(LOGS))


def gpu_selector():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    selected = visible.split(",", 1)[0].strip() if visible is not None else "0"
    return [] if selected in ("", "-1") else ["--id", selected]


def prepare():
    for path in (CACHE, LOGS, CONSOLE, RESULTS, FIGURES, HARDWARE):
        path.mkdir(parents=True, exist_ok=True)
    manifest = RESULTS / "environment.json"
    if not manifest.exists():
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
        gpu = subprocess.run(
            ["nvidia-smi", *gpu_selector(), "--query-gpu=name,driver_version,memory.total,power.limit", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        ) if shutil.which("nvidia-smi") else None
        try:
            import torch
            torch_info = {
                "version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
            }
        except (ImportError, RuntimeError) as error:
            torch_info = {"error": str(error)}
        try:
            torch_package_version = importlib.metadata.version("torch")
        except importlib.metadata.PackageNotFoundError:
            torch_package_version = None
        manifest.write_text(json.dumps({
            "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "git_commit": git.stdout.strip() if git.returncode == 0 else None,
            "git_status": subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.splitlines(),
            "python": sys.version,
            "platform": platform.platform(),
            "torch_package_version": torch_package_version,
            "torch": torch_info,
            "nvidia_smi": gpu.stdout.splitlines() if gpu and gpu.returncode == 0 else None,
            "settings": {name: os.environ.get(name) for name in ("DEPTH", "VOCAB_SIZE", "DEVICE_TYPE", "DEVICE_BATCH_SIZE", "MAX_SEQ_LEN", "EXPERIMENT_TAG", "CUDA_VISIBLE_DEVICES")},
        }, indent=2) + "\n", encoding="utf-8")


def environment():
    values = os.environ.copy()
    values.setdefault("NANOCHAT_BASE_DIR", str(CACHE))
    values.setdefault("NANOCHAT_LOG_DIR", str(LOGS))
    values.setdefault("PYTHONUNBUFFERED", "1")
    values.setdefault("OMP_NUM_THREADS", "1")
    return values


def value(name, default, cast=str):
    return cast(os.environ.get(name, default))


def tag():
    return value("EXPERIMENT_TAG", f"d{value('DEPTH', 2, int)}v{value('VOCAB_SIZE', 32768, int)}")


def append_event(record):
    prepare()
    record = {"time_utc": dt.datetime.now(dt.timezone.utc).isoformat(), **record}
    with (RESULTS / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path, data):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _gpu_samples(path, stopped):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_utc", "gpu", "name", "power_w", "utilization_percent", "memory_mib"])
        while not stopped.is_set():
            result = subprocess.run(
                ["nvidia-smi", *gpu_selector(), "--query-gpu=name,power.draw,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=False,
                timeout=4,
            )
            if result.returncode == 0:
                stamp = dt.datetime.now(dt.timezone.utc).isoformat()
                for index, row in enumerate(csv.reader(result.stdout.splitlines())):
                    if len(row) == 4:
                        writer.writerow([stamp, index, *[field.strip() for field in row]])
                stream.flush()
            stopped.wait(5)


def run_command(name, command, sample_gpu=False):
    prepare()
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = CONSOLE / f"{name}-{stamp}.log"
    command = [str(part) for part in command]
    env = environment()
    append_event({"event": "start", "stage": name, "command": command, "log": str(log_path)})
    stop = threading.Event()
    monitor = None
    sample_path = HARDWARE / f"{name}-{stamp}.csv"
    if sample_gpu and shutil.which("nvidia-smi"):
        monitor = threading.Thread(target=_gpu_samples, args=(sample_path, stop), daemon=True)
        monitor.start()
    started = time.monotonic()
    try:
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"START_UTC {dt.datetime.now(dt.timezone.utc).isoformat()}\n")
            log.write(f"COMMAND {shlex.join(command)}\n")
            log.flush()
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
            code = process.wait()
            elapsed = time.monotonic() - started
            log.write(f"END_UTC {dt.datetime.now(dt.timezone.utc).isoformat()}\n")
            log.write(f"EXIT_CODE {code}\nELAPSED_SECONDS {elapsed:.3f}\n")
        if code:
            append_event({"event": "failure", "stage": name, "exit_code": code, "elapsed_seconds": elapsed, "log": str(log_path)})
            raise subprocess.CalledProcessError(code, command)
        append_event({"event": "success", "stage": name, "elapsed_seconds": elapsed, "log": str(log_path), "gpu_samples": str(sample_path) if monitor else None})
        return log_path
    except BaseException as error:
        if not isinstance(error, subprocess.CalledProcessError):
            append_event({"event": "failure", "stage": name, "error": str(error), "log": str(log_path)})
        raise
    finally:
        if monitor:
            stop.set()
            monitor.join(timeout=10)


def python_module(module, *args):
    return [sys.executable, "-m", module, *map(str, args)]
