"""
Turn the .jsonl run logs into the figures the report needs.

Task 2 asks for training and validation bits-per-byte curves (bpb, not raw loss, because
bpb is invariant to vocabulary size and so is comparable across tokenizers). Task 3 asks
for a table of benchmark scores per stage. Both come out of the same log files.

Run:
    python -m assignment.plot_runs ~/.cache/nanochat/logs/*.jsonl --out assignment/figures
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a run killed mid-write can leave one torn line; ignore it
    return records


def series(records, key, x="step"):
    pts = [(r[x], r[key]) for r in records if key in r and x in r and r[key] is not None]
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts]


def meta(records, kind):
    for r in records:
        if r.get("record") == kind:
            return r
    return {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="+", help=".jsonl files (globs are fine)")
    p.add_argument("--out", default="assignment/figures")
    p.add_argument("--x", default="step", choices=["step", "total_training_flops", "total_training_time"])
    args = p.parse_args()

    paths = [p_ for pat in args.logs for p_ in sorted(glob.glob(pat))]
    if not paths:
        raise SystemExit("no log files matched")
    os.makedirs(args.out, exist_ok=True)

    fig, (ax_bpb, ax_loss) = plt.subplots(1, 2, figsize=(11, 4.2))
    summary_rows = []

    for path in paths:
        recs = read_jsonl(path)
        label = os.path.basename(path).replace(".jsonl", "")
        m = meta(recs, "model")

        xs, ys = series(recs, "val/bpb", args.x)
        if xs:
            ax_bpb.plot(xs, ys, marker="o", ms=3, label=f"{label} (val)")
        xs, ys = series(recs, "train/loss", args.x)
        if xs:
            ax_loss.plot(xs, ys, lw=1, label=label)

        s = meta(recs, "summary")
        if s or m:
            summary_rows.append({
                "run": label,
                "depth": m.get("depth"),
                "n_embd": (m.get("model_config") or {}).get("n_embd"),
                "n_head": (m.get("model_config") or {}).get("n_head"),
                "params": (m.get("param_counts") or {}).get("total"),
                "total_tokens": m.get("total_tokens"),
                "num_iterations": m.get("num_iterations"),
                "final_val_bpb": s.get("final_val_bpb"),
                "min_val_bpb": s.get("min_val_bpb"),
                "train_minutes": (s.get("total_training_time_s") or 0) / 60,
                "peak_mem_mib": s.get("peak_memory_mib"),
            })

    ax_bpb.set_xlabel(args.x)
    ax_bpb.set_ylabel("validation bits per byte")
    ax_bpb.set_title("Validation bpb")
    ax_bpb.grid(alpha=0.3)
    ax_bpb.legend(fontsize=7)

    ax_loss.set_xlabel(args.x)
    ax_loss.set_ylabel("training loss (EMA)")
    ax_loss.set_title("Training loss")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend(fontsize=7)

    fig.tight_layout()
    out_png = os.path.join(args.out, "curves.png")
    fig.savefig(out_png, dpi=200)
    print(f"wrote {out_png}")

    if summary_rows:
        cols = list(summary_rows[0])
        print("\n| " + " | ".join(cols) + " |")
        print("|" + "---|" * len(cols))
        for row in summary_rows:
            cells = []
            for c in cols:
                v = row[c]
                cells.append(f"{v:,.4f}" if isinstance(v, float) else (f"{v:,}" if isinstance(v, int) else str(v)))
            print("| " + " | ".join(cells) + " |")
        with open(os.path.join(args.out, "summary.json"), "w") as f:
            json.dump(summary_rows, f, indent=2)
        print(f"\nwrote {os.path.join(args.out, 'summary.json')}")


if __name__ == "__main__":
    main()
