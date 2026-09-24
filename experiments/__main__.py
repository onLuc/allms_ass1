import argparse

from experiments.common import append_event, python_module, run_command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=[f"task_{index}" for index in range(5)])
    args, rest = parser.parse_known_args()
    append_event({"event": "start", "stage": "package", "task": args.task})
    run_command(args.task, python_module(f"experiments.{args.task}", *rest))


if __name__ == "__main__":
    main()
