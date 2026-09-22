"""
Task 1 evidence generator: compare two trained tokenizers head to head.

Produces, as JSON and as a markdown table you can paste into the report:
  - compression ratio (bytes/token and tokens/char) per text domain
  - sequence length for one fixed text sample under each tokenizer
  - the downstream cost of the vocabulary: embedding + lm_head parameters it implies
  - concrete failure cases: numbers, source code, non-English text, whitespace

Run (after training both tokenizers with --out-dir):
    python -m assignment.tok_compare \
        --tokenizers ~/.cache/nanochat/tokenizer_8192 ~/.cache/nanochat/tokenizer_32768 \
        --out assignment/results/tok_compare.json
"""

import argparse
import json
import os

from nanochat.tokenizer import RustBPETokenizer
from nanochat.dataset import parquets_iter_batched

# Probe texts. English comes from the real pretraining distribution (loaded below);
# the rest are deliberately chosen to expose where a BPE vocabulary is badly matched.
PROBES = {
    "numbers": "The population reached 1234567 in 2024, up 3.7% from 1190432, a gain of 44135 people.",
    "python": (
        "def fibonacci(n: int) -> list[int]:\n"
        "    seq = [0, 1]\n"
        "    while len(seq) < n:\n"
        "        seq.append(seq[-1] + seq[-2])\n"
        "    return seq[:n]\n"
    ),
    "dutch": "Het regent vandaag in Leiden, dus ik neem de trein naar het Snellius-gebouw.",
    "chinese": "今天天气很好，我们去公园散步吧。",
    "whitespace": "a" + " " * 40 + "b\n\n\n\t\tindented\n",
    "rare_words": "The otorhinolaryngologist diagnosed pneumonoultramicroscopicsilicovolcanoconiosis.",
    "urls": "See https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main/shard_00042.parquet",
}


def load_english_sample(max_chars=200_000):
    """Pull a fixed English sample out of the ClimbMix validation shard."""
    text = []
    total = 0
    for batch in parquets_iter_batched(split="val"):
        for doc in batch:
            text.append(doc)
            total += len(doc)
            if total >= max_chars:
                return "\n\n".join(text)[:max_chars]
    return "\n\n".join(text)[:max_chars]


def measure(tok, text):
    ids = tok.encode(text)
    n_bytes = len(text.encode("utf-8"))
    n_tokens = len(ids)
    return {
        "chars": len(text),
        "bytes": n_bytes,
        "tokens": n_tokens,
        "bytes_per_token": n_bytes / n_tokens if n_tokens else 0.0,
        "tokens_per_char": n_tokens / len(text) if text else 0.0,
    }


def embedding_cost(vocab_size, n_embd):
    """What this vocabulary costs in parameters at a given model width.

    nanochat has three vocab-sized matrices: wte, lm_head and (on alternating layers)
    the value embeddings. At small depths these dominate the parameter count entirely.
    """
    return {
        "wte": vocab_size * n_embd,
        "lm_head": vocab_size * n_embd,
        "per_value_embed": vocab_size * n_embd,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizers", nargs="+", required=True, help="directories saved by scripts/tok_train.py --out-dir")
    p.add_argument("--out", type=str, default="assignment/results/tok_compare.json")
    p.add_argument("--english-chars", type=int, default=200_000)
    p.add_argument("--n-embd", type=int, default=128, help="model width, for the embedding-cost column")
    args = p.parse_args()

    print("Loading English sample from the ClimbMix validation shard...")
    probes = dict(PROBES)
    probes["english"] = load_english_sample(args.english_chars)

    results = {}
    for d in args.tokenizers:
        tok = RustBPETokenizer.from_directory(d)
        name = os.path.basename(os.path.normpath(d))
        vocab_size = tok.get_vocab_size()
        results[name] = {
            "dir": d,
            "vocab_size": vocab_size,
            "embedding_cost_at_n_embd": {str(args.n_embd): embedding_cost(vocab_size, args.n_embd)},
            "probes": {k: measure(tok, v) for k, v in probes.items()},
        }
        print(f"  {name}: vocab_size={vocab_size:,}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    # markdown table, bytes/token (higher = better compression)
    names = list(results)
    print("\n### Compression: bytes per token (higher is better)\n")
    print("| domain | " + " | ".join(names) + " | ratio |")
    print("|---|" + "---|" * (len(names) + 1))
    for probe in probes:
        vals = [results[n]["probes"][probe]["bytes_per_token"] for n in names]
        ratio = vals[-1] / vals[0] if vals[0] else float("nan")
        print(f"| {probe} | " + " | ".join(f"{v:.2f}" for v in vals) + f" | {ratio:.2f}x |")

    print("\n### Sequence length for the same text (tokens)\n")
    print("| domain | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for probe in probes:
        vals = [results[n]["probes"][probe]["tokens"] for n in names]
        print(f"| {probe} | " + " | ".join(f"{v:,}" for v in vals) + " |")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
