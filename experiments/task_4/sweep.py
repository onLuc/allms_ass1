import argparse
import csv
import json

from nanochat.checkpoint_manager import load_model
from nanochat.common import compute_cleanup, compute_init
from nanochat.engine import Engine

from experiments.common import RESULTS, prepare, read_json


PROMPTS = (
    "The sky appears blue because",
    "A clear explanation of photosynthesis is",
    "The main difference between mitosis and meiosis is",
    "A simple recipe for pancakes begins with",
    "If a book costs 12 euros and I pay with 20 euros, my change is",
)


def generate(engine, tokenizer, prompt, temperature, seed, max_tokens):
    ids = [
        tokenizer.get_bos_token_id(),
        tokenizer.encode_special("<|user_start|>"),
        *tokenizer.encode(prompt),
        tokenizer.encode_special("<|user_end|>"),
        tokenizer.encode_special("<|assistant_start|>"),
    ]
    end = tokenizer.encode_special("<|assistant_end|>")
    tokens = []
    for column, _ in engine.generate(ids, num_samples=1, max_tokens=max_tokens, temperature=temperature, top_k=50, seed=seed):
        token = int(column[0].item())
        if token == end:
            break
        tokens.append(token)
    return tokenizer.decode(tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["base", "mid", "sft"], default="sft")
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--device-type", choices=["cuda", "cpu", "mps"], default="cuda")
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    prepare()
    _, _, _, _, device = compute_init(args.device_type)
    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag)
    engine = Engine(model, tokenizer)
    rows = []
    for index, prompt in enumerate(PROMPTS):
        for temperature in (0.1, 0.7, 1.5):
            rows.append({
                "model_tag": args.model_tag,
                "source": args.source,
                "checkpoint_step": meta["step"],
                "prompt_id": index + 1,
                "prompt": prompt,
                "temperature": temperature,
                "top_k": 50,
                "seed": 42 + index,
                "max_tokens": args.max_tokens,
                "response": generate(engine, tokenizer, prompt, temperature, 42 + index, args.max_tokens),
            })
    path = RESULTS / "task_4_temperature_sweep.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["| Prompt | Temperature | Response |", "|---:|---:|---|"]
    for row in rows:
        prompt = row["prompt"].replace("|", "\\|")
        response = row["response"].replace("|", "\\|").replace("\n", "<br>")
        lines.append(f"| {row['prompt_id']}: {prompt} | {row['temperature']} | {response} |")
    (RESULTS / "task_4_temperature_sweep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sizing_path = RESULTS / "task_2_scaling.json"
    parameters = read_json(sizing_path).get("trainable_parameters", "") if sizing_path.exists() else ""
    comparison_path = RESULTS / "task_4_published_comparison_template.csv"
    with comparison_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "parameters", "ARC-Easy", "ARC-Easy source", "ARC-Challenge", "ARC-Challenge source", "GSM8K", "GSM8K source"])
        writer.writerow([f"Nanochat {args.model_tag}", parameters, "", "", "", "", "", ""])
        smol_url = "https://huggingface.co/HuggingFaceTB/SmolLM2-360M"
        qwen_url = "https://huggingface.co/Qwen/Qwen2.5-7B"
        writer.writerow(["HuggingFaceTB/SmolLM2-360M", "360M", "", smol_url, "", smol_url, "", smol_url])
        writer.writerow(["Qwen/Qwen2.5-7B", "7B", "", qwen_url, "", qwen_url, "", qwen_url])
    print(f"Wrote {path}, {RESULTS / 'task_4_temperature_sweep.md'} and {comparison_path}")
    compute_cleanup()


if __name__ == "__main__":
    main()
