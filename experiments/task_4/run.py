import argparse

from experiments.common import append_event, python_module, run_command, tag, value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["base", "mid", "sft"], default="sft")
    parser.add_argument("--device-type", choices=["cuda", "cpu", "mps"], default="cuda")
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    model_tag = tag()
    device = args.device_type or value("DEVICE_TYPE", "cuda")
    append_event({"event": "start", "stage": "task_4", "source": args.source, "tag": model_tag})
    run_command("task4-temperature-sweep", python_module(
        "experiments.task_4.sweep",
        "--source", args.source,
        "--model-tag", model_tag,
        "--device-type", device,
        "--max-tokens", args.max_tokens,
    ), sample_gpu=device == "cuda")
    append_event({"event": "success", "stage": "task_4", "source": args.source, "tag": model_tag})


if __name__ == "__main__":
    main()
