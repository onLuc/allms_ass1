import argparse
import csv
import re
import subprocess

from experiments.common import CACHE, RESULTS, ROOT, append_event, python_module, run_command, tag, value


TASKS = ("ARC-Easy", "ARC-Challenge", "GSM8K")


def evaluate(source, model_tag, device, stage):
    rows = []
    for task_name in TASKS:
        log_path = run_command(f"task3-eval-{stage}-{task_name.lower()}", python_module(
            "scripts.chat_eval",
            "--device-type", device,
            "--source", source,
            "--task-name", task_name,
            "--model-tag", model_tag,
            "--batch-size", 1,
            "--temperature", 0,
            "--num-samples", 1,
            "--max-new-tokens", 512,
        ), sample_gpu=device == "cuda")
        text = log_path.read_text(encoding="utf-8")
        found = re.search(r"Final: (\d+)/(\d+) \(([0-9.]+)%\)", text)
        if not found:
            raise RuntimeError(f"Could not parse {task_name} result from {log_path}")
        correct, total, accuracy = found.groups()
        rows.append({
            "stage": stage,
            "source": source,
            "task": task_name,
            "correct": int(correct),
            "total": int(total),
            "accuracy_percent": float(accuracy),
            "random_baseline_percent": 25.0 if task_name.startswith("ARC-") else 0.0,
            "log": str(log_path),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmlu-epochs", type=int, default=3)
    parser.add_argument("--gsm8k-epochs", type=int, default=4)
    parser.add_argument("--eval-tokens", type=int, default=131072)
    parser.add_argument("--skip-inspection", action="store_true")
    args = parser.parse_args()
    batch = value("DEVICE_BATCH_SIZE", 2, int)
    sequence = value("MAX_SEQ_LEN", 2048, int)
    device = value("DEVICE_TYPE", "cuda")
    model_tag = tag()
    append_event({"event": "start", "stage": "task_3", "tag": model_tag})
    mid_dir = CACHE / "midtrain_checkpoints" / model_tag
    sft_dir = CACHE / "chatsft_checkpoints" / model_tag
    if mid_dir.exists() or sft_dir.exists():
        raise FileExistsError(f"A mid-training or SFT checkpoint already exists for {model_tag}; use a new run root or EXPERIMENT_TAG")
    if not list((CACHE / "base_checkpoints" / model_tag).glob("model_*.pt")):
        raise FileNotFoundError(f"No base model checkpoint for {model_tag}; run Task 2 first")
    diff = subprocess.run(["git", "diff", "upstream/master", "--", "scripts/chat_sft.py", "nanochat/checkpoint_manager.py"], cwd=ROOT, capture_output=True, text=True, check=False)
    diff_path = RESULTS / "task_3_upstream_diff.patch"
    diff_text = diff.stdout if diff.returncode == 0 else f"Could not compare against upstream/master: {diff.stderr.strip()}\n"
    diff_path.write_text(diff_text, encoding="utf-8")
    if not args.skip_inspection:
        run_command("task3-data-inspection", python_module("experiments.task_3.inspect_data", "--mmlu-epochs", args.mmlu_epochs, "--gsm8k-epochs", args.gsm8k_epochs))
    result_rows = evaluate("base", model_tag, device, "base")
    run_command("task3-mid-train", python_module(
        "scripts.chat_sft",
        "--device-type", device,
        "--stage", "mid",
        "--source", "base",
        "--model-tag", model_tag,
        "--out-tag", model_tag,
        "--max-seq-len", sequence,
        "--device-batch-size", batch,
        "--total-batch-size", 131072,
        "--mmlu-epochs", args.mmlu_epochs,
        "--gsm8k-epochs", args.gsm8k_epochs,
        "--num-iterations", -1,
        "--eval-every", 25,
        "--eval-tokens", args.eval_tokens,
        "--chatcore-every", -1,
        "--log-every", 5,
    ), sample_gpu=device == "cuda")
    result_rows.extend(evaluate("mid", model_tag, device, "mid"))
    run_command("task3-sft-train", python_module(
        "scripts.chat_sft",
        "--device-type", device,
        "--stage", "sft",
        "--source", "mid",
        "--model-tag", model_tag,
        "--out-tag", model_tag,
        "--max-seq-len", sequence,
        "--device-batch-size", batch,
        "--total-batch-size", 131072,
        "--num-iterations", -1,
        "--eval-every", 25,
        "--eval-tokens", args.eval_tokens,
        "--chatcore-every", -1,
        "--log-every", 5,
    ), sample_gpu=device == "cuda")
    result_rows.extend(evaluate("sft", model_tag, device, "sft"))
    result_path = RESULTS / "task_3_benchmarks.csv"
    with result_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=result_rows[0].keys())
        writer.writeheader()
        writer.writerows(result_rows)
    table = ["| Stage | Benchmark | Correct | Total | Accuracy | Random baseline |", "|---|---|---:|---:|---:|---:|"]
    for item in result_rows:
        table.append(f"| {item['stage']} | {item['task']} | {item['correct']} | {item['total']} | {item['accuracy_percent']:.2f}% | {item['random_baseline_percent']:.0f}% |")
    (RESULTS / "task_3_benchmarks.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    append_event({"event": "success", "stage": "task_3", "tag": model_tag, "benchmark_rows": len(result_rows)})


if __name__ == "__main__":
    main()
