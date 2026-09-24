import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not rows:
        raise ValueError("No checkpoint evaluations were supplied")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with (out.with_suffix(".csv")).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["step", "train_bpb", "validation_bpb"])
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot([row["step"] for row in rows], [row["train_bpb"] for row in rows], marker="o", label="Training subset")
    ax.plot([row["step"] for row in rows], [row["validation_bpb"] for row in rows], marker="o", label="Validation subset")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Bits per byte")
    ax.set_title("Pretraining bits per byte on fixed CLIMBMix subsets")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"Wrote {out} and {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
