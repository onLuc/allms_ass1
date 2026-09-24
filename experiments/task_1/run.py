import argparse
import json
import os
import shutil
from pathlib import Path

from nanochat.tokenizer import RustBPETokenizer

from experiments.common import CACHE, RESULTS, append_event, python_module, run_command, tag, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-bytes", type=int, default=500_000_000)
    parser.add_argument("--shards", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--merge-sample-bytes", type=int, default=5_000_000)
    parser.add_argument("--word", default=" counterproductive")
    args = parser.parse_args()
    append_event({"event": "start", "stage": "task_1", "sample_bytes": args.sample_bytes})
    run_command("task1-download", python_module("nanochat.dataset", "-n", args.shards, "-w", 2))
    sizes = (8192, 32768)
    expected = [CACHE / f"tokenizer_{size}" for size in sizes]
    active = CACHE / "tokenizer"
    if args.force and active.exists():
        raise RuntimeError("A tokenizer is already frozen for this cache; choose a fresh NANOCHAT_BASE_DIR before retraining")
    if args.force and any((CACHE / folder).exists() for folder in ("base_checkpoints", "midtrain_checkpoints", "chatsft_checkpoints")):
        raise RuntimeError("Checkpoints already exist in this cache; choose a fresh NANOCHAT_BASE_DIR before retraining tokenizers")
    missing = [size for size, path in zip(sizes, expected) if not (path / "token_bytes.pt").is_file()]
    if args.force:
        missing = list(sizes)
    if missing:
        run_command("task1-tokenizers", python_module("experiments.task_1.train_tokenizers", "--vocab-sizes", *missing, "--sample-bytes", args.sample_bytes, *( ["--force"] if args.force else [] )))
    selected = int(os.environ.get("VOCAB_SIZE", "32768"))
    if selected not in sizes:
        raise ValueError("VOCAB_SIZE must be 8192 or 32768 for the assignment")
    if active.exists():
        active_size = RustBPETokenizer.from_directory(str(active)).get_vocab_size()
        if active_size != selected:
            raise RuntimeError(f"Active tokenizer has vocab {active_size}; create a fresh NANOCHAT_BASE_DIR to use vocab {selected} safely")
    else:
        shutil.copytree(CACHE / f"tokenizer_{selected}", active)
    run_command("task1-compare", python_module("assignment.tok_compare", "--tokenizers", *expected, "--out", RESULTS / "task_1_tokenizer_comparison.json", "--english-chars", 200000, "--n-embd", 128))
    comparison = json.loads((RESULTS / "task_1_tokenizer_comparison.json").read_text(encoding="utf-8"))
    names = list(comparison)
    probes = next(iter(comparison.values()))["probes"]
    lines = ["| Measurement | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for domain in probes:
        values = [comparison[name]["probes"][domain] for name in names]
        lines.append(f"| {domain} tokens | " + " | ".join(f"{item['tokens']:,}" for item in values) + " |")
        lines.append(f"| {domain} bytes/token | " + " | ".join(f"{item['bytes_per_token']:.3f}" for item in values) + " |")
        lines.append(f"| {domain} tokens/character | " + " | ".join(f"{item['tokens_per_char']:.4f}" for item in values) + " |")
    (RESULTS / "task_1_tokenizer_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    run_command("task1-model-sizing", python_module("assignment.model_sizing", "--depths", 2, "--vocab-sizes", *sizes, "--json-out", RESULTS / "task_1_vocab_model_sizing.json"))
    run_command("task1-merge-tree", python_module("experiments.task_1.merge_tree", "--vocab-size", selected, "--sample-bytes", args.merge_sample_bytes, "--word", args.word))
    write_json(RESULTS / "task_1_manifest.json", {
        "sample_bytes_per_tokenizer": args.sample_bytes,
        "vocab_sizes": sizes,
        "selected_vocab_size": selected,
        "active_tokenizer": str(active),
        "run_tag": tag(),
        "outputs": ["task_1_tokenizer_comparison.json", "task_1_tokenizer_comparison.md", "task_1_vocab_model_sizing.json", "task_1_merge_tree.txt", "task_1_merge_tree.dot", "task_1_merge_tree.json"],
    })
    append_event({"event": "success", "stage": "task_1", "vocab_sizes": sizes, "selected_vocab_size": selected})


if __name__ == "__main__":
    main()
