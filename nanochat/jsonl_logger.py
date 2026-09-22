"""
Structured .jsonl logging for nanochat runs.

Why this exists
---------------
nanochat logs metrics in two places: pretty-printed lines on stdout (nice to watch,
painful to parse) and wandb (nice to plot, needs an account and a network). For a
course assignment you want a third thing: a plain file on disk, one JSON object per
line, that you can re-read weeks later to make the plots for your report.

Every metric in base_train.py / chat_sft.py already flows through a single object
(`wandb_run`) with a `.log(dict)` interface. So we do not need to touch the training
loop at all: we just wrap that object. Every record is also given a wall-clock
timestamp and a `record` kind so the file stays self-describing.

Usage (already wired into the training scripts):

    from nanochat.jsonl_logger import get_run_logger
    wandb_run = get_run_logger(run=args.run, project="nanochat",
                               config=user_config, tag="base_d2",
                               master_process=master_process)
    wandb_run.log_meta("model", {...})   # one-off facts (config, param counts, ...)
    wandb_run.log({"step": 0, "val/bpb": 1.23})   # time series, as before
    wandb_run.finish()

Reading it back:

    import json
    records = [json.loads(l) for l in open(path)]
    curve = [(r["step"], r["val/bpb"]) for r in records if "val/bpb" in r]
"""

import os
import json
import time
import datetime

from nanochat.common import get_base_dir, DummyWandb, print0


class JsonlLogger:
    """
    wandb-compatible logger (.log / .finish) that tees every record into a .jsonl file.

    Only the master process should ever hold a writing instance; other ranks get one
    with path=None, which makes every method a no-op.
    """

    def __init__(self, path=None, inner=None):
        self.path = path
        self.inner = inner if inner is not None else DummyWandb()
        self.t_start = time.time()
        if self.path is not None:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            # line-buffered append: survives a crash / Ctrl-C without losing the run
            self.f = open(self.path, "a", buffering=1, encoding="utf-8")
        else:
            self.f = None

    def _write(self, record):
        if self.f is None:
            return
        record = dict(record)
        record.setdefault("record", "metric")
        record["wall_time"] = round(time.time() - self.t_start, 3)
        record["timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")
        # default=str so that a stray tensor/np scalar never kills a 2-hour run
        self.f.write(json.dumps(record, default=str) + "\n")

    def log(self, data, *args, **kwargs):
        """Time-series record. Mirrors wandb.log."""
        self._write(data)
        self.inner.log(data, *args, **kwargs)

    def log_meta(self, kind, data):
        """
        One-off record that is not a time series: run config, model shape, parameter
        counts, final results. These are the numbers you need for the report tables
        and they are exactly the ones you forget to write down.
        """
        self._write({"record": kind, **data})
        # wandb keeps these in the run config rather than the metric stream
        try:
            self.inner.config.update({kind: data}, allow_val_change=True)
        except Exception:
            pass

    def finish(self):
        if self.f is not None:
            self._write({"record": "finish"})
            self.f.close()
            self.f = None
        self.inner.finish()


def get_run_logger(run, project, config, tag, master_process=True, log_dir=None):
    """
    Build the logger used by the training scripts.

    - Always writes <base_dir>/logs/<tag>_<timestamp>.jsonl on the master process.
    - Additionally logs to wandb if `run` is not "dummy" (unchanged nanochat behaviour).
    """
    if not master_process:
        return JsonlLogger(path=None, inner=DummyWandb())

    inner = DummyWandb()
    if run != "dummy":
        import wandb
        inner = wandb.init(project=project, name=run, config=config)

    log_dir = log_dir or os.path.join(get_base_dir(), "logs")
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(log_dir, f"{tag}_{stamp}.jsonl")
    logger = JsonlLogger(path=path, inner=inner)
    logger.log_meta("run_config", {"tag": tag, "project": project, "config": config})
    print0(f"[jsonl_logger] writing structured logs to: {path}")
    return logger
