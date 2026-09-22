"""
Work out, on CPU and without touching any data, exactly what nanochat will build for a
given --depth, and what training horizon it will pick.

This answers Task 2's "state the number of layers, attention heads, embedding dimension,
the total trainable parameter count, and the number of total tokens the model will be
trained on" without having to start (and wait for) a training run, and it lets you see
how the answers move with vocab size, which is the bridge from Task 1 to Task 2.

Run:
    python -m assignment.model_sizing
    python -m assignment.model_sizing --depths 2 4 --vocab-sizes 8192 32768
"""

import argparse
import json
import math

import torch

from nanochat.gpt import GPT, GPTConfig


def build(depth, vocab_size, aspect_ratio=64, head_dim=128, max_seq_len=2048, window_pattern="L"):
    """Mirror of build_model_meta() in scripts/base_train.py."""
    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    num_heads = model_dim // head_dim
    config = GPTConfig(
        sequence_len=max_seq_len, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=window_pattern,
    )
    with torch.device("meta"):
        model = GPT(config)
    return model, config


def scaling_params_of(model):
    c = model.num_scaling_params()
    return c["transformer_matrices"] + c["lm_head"]


def describe(depth, vocab_size, target_param_data_ratio=12.0, **kw):
    model, config = build(depth, vocab_size, **kw)
    counts = model.num_scaling_params()
    # base_train.py drives the horizon off transformer matrices + lm_head only
    scaling_params = counts["transformer_matrices"] + counts["lm_head"]
    target_tokens = int(target_param_data_ratio * scaling_params)
    flops_per_token = model.estimate_flops()

    # Mirror of base_train.py's auto batch size: Bopt grows as D^0.383 relative to the
    # d12 reference run (Power Lines, arXiv:2505.13738), clamped to a power of two.
    d12_ref, _ = build(12, vocab_size, **kw)
    d_ref = target_param_data_ratio * scaling_params_of(d12_ref)
    b_ref = 2 ** 19
    predicted_batch = b_ref * (target_tokens / d_ref) ** 0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch))
    num_iterations = target_tokens // total_batch_size

    return {
        "total_batch_size": total_batch_size,
        "num_iterations": num_iterations,
        "depth": depth,
        "vocab_size": vocab_size,
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "head_dim": config.n_embd // config.n_head,
        "n_embd": config.n_embd,
        "params_total": counts["total"],
        "params_embedding": counts["wte"] + counts["value_embeds"] + counts["lm_head"],
        "params_transformer_matrices": counts["transformer_matrices"],
        "embedding_fraction": (counts["wte"] + counts["value_embeds"] + counts["lm_head"]) / counts["total"],
        "scaling_params": scaling_params,
        "target_tokens": target_tokens,
        "flops_per_token": flops_per_token,
        "total_train_flops": flops_per_token * target_tokens,
        "param_counts": counts,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--depths", type=int, nargs="+", default=[2, 4, 6, 12])
    p.add_argument("--vocab-sizes", type=int, nargs="+", default=[8192, 32768])
    p.add_argument("--target-param-data-ratio", type=float, default=12.0)
    p.add_argument("--json-out", type=str, default=None)
    args = p.parse_args()

    rows = [describe(d, v, args.target_param_data_ratio)
            for v in args.vocab_sizes for d in args.depths]

    hdr = f"{'depth':>5} {'vocab':>7} {'heads':>5} {'n_embd':>6} " \
          f"{'params':>12} {'emb%':>5} {'tokens':>13} {'batch':>8} {'steps':>7} " \
          f"{'FLOPs/tok':>10} {'train FLOPs':>12}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['depth']:>5} {r['vocab_size']:>7} {r['n_head']:>5} {r['n_embd']:>6} "
              f"{r['params_total']:>12,} {100*r['embedding_fraction']:>4.0f}% {r['target_tokens']:>13,} "
              f"{r['total_batch_size']:>8,} {r['num_iterations']:>7,} "
              f"{r['flops_per_token']:>10.2e} {r['total_train_flops']:>12.2e}")

    # Wall-clock is FLOPs / (peak FLOPs x MFU). A GTX 1070 has no bf16 tensor cores and
    # no Triton, so expect the low end of this range; an A100 the high end.
    print("\nEstimated pretraining wall-clock (hours):")
    devices = [("GTX 1070 (6.5 TFLOP/s fp32)", 6.5e12, [0.05, 0.10, 0.20]),
               ("A100 (312 TFLOP/s bf16)", 312e12, [0.20, 0.40])]
    for name, peak, mfus in devices:
        for r in rows:
            if r["vocab_size"] != 32768:
                continue
            times = "  ".join(f"MFU {int(100*m):>2}%: {r['total_train_flops']/(peak*m)/3600:6.2f}h" for m in mfus)
            print(f"  d{r['depth']:<2} {name:<28} {times}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
