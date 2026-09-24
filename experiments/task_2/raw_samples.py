import argparse
import json

from nanochat.checkpoint_manager import load_model
from nanochat.common import compute_cleanup, compute_init
from nanochat.engine import Engine

from experiments.common import RESULTS, prepare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--device-type", choices=["cuda", "cpu", "mps"], default="cuda")
    args = parser.parse_args()
    prepare()
    _, _, _, _, device = compute_init(args.device_type)
    model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag)
    prompt = tokenizer("", prepend="<|bos|>")
    completions, _ = Engine(model, tokenizer).generate_batch(prompt, num_samples=5, max_tokens=128, temperature=1.0, top_k=50, seed=42)
    rows = [{"sample": index + 1, "model_tag": args.model_tag, "checkpoint_step": meta["step"], "prompt": "", "temperature": 1.0, "top_k": 50, "seed": 42, "completion": tokenizer.decode(tokens[len(prompt):])} for index, tokens in enumerate(completions)]
    path = RESULTS / "task_2_raw_completions.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    text = "\n\n".join(f"Sample {row['sample']}\n{row['completion']}" for row in rows)
    (RESULTS / "task_2_raw_completions.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    compute_cleanup()


if __name__ == "__main__":
    main()
