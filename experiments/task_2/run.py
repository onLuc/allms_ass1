import argparse
import glob
import json
import re
from pathlib import Path

from nanochat.tokenizer import RustBPETokenizer

from experiments.common import CACHE, FIGURES, LOGS, RESULTS, append_event, python_module, read_json, run_command, tag, value, write_json


def read_records(path):
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-tokens", type=int, default=131072)
    parser.add_argument("--sample-every", type=int, default=50)
    parser.add_argument("--bpb-tokens", type=int, default=131072)
    args = parser.parse_args()
    depth = value("DEPTH", 2, int)
    vocab = value("VOCAB_SIZE", 32768, int)
    batch = value("DEVICE_BATCH_SIZE", 2, int)
    sequence = value("MAX_SEQ_LEN", 2048, int)
    device = value("DEVICE_TYPE", "cuda")
    model_tag = tag()
    append_event({"event": "start", "stage": "task_2", "depth": depth, "vocab_size": vocab, "tag": model_tag})
    checkpoint_dir = CACHE / "base_checkpoints" / model_tag
    if checkpoint_dir.exists():
        raise FileExistsError(f"{checkpoint_dir} already exists; use a new run root or EXPERIMENT_TAG")
    active_tokenizer = CACHE / "tokenizer"
    if not active_tokenizer.is_dir():
        raise FileNotFoundError("Run Task 1 first so the chosen tokenizer is trained and frozen")
    active_vocab = RustBPETokenizer.from_directory(str(active_tokenizer)).get_vocab_size()
    if active_vocab != vocab:
        raise RuntimeError(f"Active tokenizer vocab is {active_vocab}, but VOCAB_SIZE={vocab}")
    sizing_path = RESULTS / "task_2_model_sizing.json"
    run_command("task2-sizing", python_module("assignment.model_sizing", "--depths", depth, "--vocab-sizes", vocab, "--target-param-data-ratio", 12, "--json-out", sizing_path))
    run_command("task2-base-train", python_module(
        "scripts.base_train",
        "--device-type", device,
        "--depth", depth,
        "--model-tag", model_tag,
        "--window-pattern", "L",
        "--max-seq-len", sequence,
        "--device-batch-size", batch,
        "--target-param-data-ratio", 12,
        "--eval-every", args.eval_every,
        "--eval-tokens", args.eval_tokens,
        "--log-every", 5,
        "--sample-every", args.sample_every,
        "--save-every", args.save_every,
        "--core-metric-every", -1,
    ), sample_gpu=device == "cuda")
    checkpoints = sorted(checkpoint_dir.glob("model_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No base checkpoints were saved in {checkpoint_dir}")
    rows = []
    for checkpoint in checkpoints:
        step = int(re.search(r"model_(\d+)\.pt$", checkpoint.name).group(1))
        log_path = run_command(f"task2-bpb-{step:06d}", python_module(
            "scripts.base_eval",
            "--device-type", device,
            "--eval", "bpb",
            "--model-tag", model_tag,
            "--step", step,
            "--device-batch-size", batch,
            "--split-tokens", args.bpb_tokens,
        ), sample_gpu=device == "cuda")
        text = log_path.read_text(encoding="utf-8")
        values = {split: re.search(rf"^{split} bpb: ([0-9.eE+-]+)", text, flags=re.MULTILINE) for split in ("train", "val")}
        if not all(values.values()):
            raise RuntimeError(f"Could not parse train/validation BPB from {log_path}")
        rows.append({"step": step, "train_bpb": float(values["train"].group(1)), "validation_bpb": float(values["val"].group(1))})
    rows = sorted({row["step"]: row for row in rows}.values(), key=lambda row: row["step"])
    bpb_path = RESULTS / "task_2_bpb.json"
    write_json(bpb_path, rows)
    run_command("task2-plot", python_module("experiments.task_2.plot", "--input", bpb_path, "--output", FIGURES / "task_2_bpb.png"))
    run_command("task2-raw-samples", python_module("experiments.task_2.raw_samples", "--device-type", device, "--model-tag", model_tag), sample_gpu=device == "cuda")
    patterns = glob.glob(str(LOGS / f"{model_tag}_*.jsonl"))
    if patterns:
        records = read_records(max(patterns, key=lambda path: Path(path).stat().st_mtime))
        model_record = next((record for record in records if record.get("record") == "model"), {})
    else:
        model_record = {}
    sizing = read_json(sizing_path)
    item = sizing[0] if isinstance(sizing, list) else sizing
    params = (model_record.get("param_counts") or {}).get("total", item.get("params_total"))
    scaling_params = model_record.get("num_scaling_params", item.get("scaling_params"))
    actual_tokens = model_record.get("total_tokens")
    write_json(RESULTS / "task_2_scaling.json", {
        "depth": depth,
        "vocab_size": vocab,
        "layers": (model_record.get("model_config") or {}).get("n_layer", item.get("n_layer")),
        "attention_heads": (model_record.get("model_config") or {}).get("n_head", item.get("n_head")),
        "embedding_dimension": (model_record.get("model_config") or {}).get("n_embd", item.get("n_embd")),
        "trainable_parameters": params,
        "nanochat_scaling_parameters_excluding_embeddings": scaling_params,
        "planned_tokens_from_training_log": actual_tokens,
        "chinchilla_20x_total_parameter_reference_tokens": 20 * params if params else None,
        "nanochat_default_ratio": 12,
        "observed_tokens_per_scaling_parameter": model_record.get("tokens_per_scaling_param"),
        "model_log_record": model_record,
    })
    append_event({"event": "success", "stage": "task_2", "tag": model_tag, "checkpoint_count": len(checkpoints)})


if __name__ == "__main__":
    main()
