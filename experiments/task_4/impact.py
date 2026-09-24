import argparse
import csv
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path

from experiments.common import HARDWARE, RESULTS, prepare


def samples(paths):
    groups = defaultdict(list)
    for path in paths:
        with Path(path).open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                try:
                    timestamp = dt.datetime.fromisoformat(row["time_utc"])
                    power = float(row["power_w"])
                    if power >= 0:
                        groups[(str(path), row["gpu"], row["name"])].append((timestamp, power))
                except (KeyError, TypeError, ValueError):
                    continue
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-carbon-g-per-kwh", type=float)
    parser.add_argument("--grid-source", default="")
    parser.add_argument("--pue", type=float, default=1.0)
    args = parser.parse_args()
    prepare()
    paths = sorted(HARDWARE.glob("task2-base-train-*.csv")) + sorted(HARDWARE.glob("task3-mid-train-*.csv")) + sorted(HARDWARE.glob("task3-sft-train-*.csv"))
    if not paths:
        raise FileNotFoundError(f"No GPU power samples found under {HARDWARE}")
    if args.grid_carbon_g_per_kwh is not None and (args.grid_carbon_g_per_kwh < 0 or not args.grid_source):
        raise ValueError("Provide a non-negative grid factor and its source URL")
    if args.pue < 1:
        raise ValueError("PUE must be at least 1")
    gpu_rows = []
    total_wh = 0.0
    for (source, gpu, name), values in samples(paths).items():
        values.sort()
        energy_wh = sum((left[1] + right[1]) / 2 * (right[0] - left[0]).total_seconds() / 3600 for left, right in zip(values, values[1:]))
        total_wh += energy_wh
        gpu_rows.append({"sample_file": source, "gpu_index": gpu, "gpu_name": name, "sample_count": len(values), "mean_power_w": sum(item[1] for item in values) / len(values), "sampled_energy_wh": energy_wh})
    if not gpu_rows:
        raise ValueError("No valid GPU power readings were found in the training logs")
    result = {
        "method": "trapezoidal integration of nvidia-smi board power samples during training stages",
        "training_sample_files": [str(path) for path in paths],
        "gpu_samples": gpu_rows,
        "sampled_gpu_energy_kwh": total_wh / 1000,
        "pue_assumption": args.pue,
        "facility_energy_kwh_estimate": total_wh / 1000 * args.pue,
        "grid_carbon_intensity_g_per_kwh": args.grid_carbon_g_per_kwh,
        "grid_carbon_source": args.grid_source or None,
        "estimated_co2_kg": total_wh / 1000 * args.pue * args.grid_carbon_g_per_kwh / 1000 if args.grid_carbon_g_per_kwh is not None else None,
        "limitations": ["GPU board power is sampled at intervals and excludes host CPU and network energy", "PUE defaults to 1.0 unless changed", "Carbon intensity must correspond to the workload location and dates"],
    }
    path = RESULTS / "task_4_environmental_estimate.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
