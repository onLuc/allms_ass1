import argparse
import json

from nanochat.tokenizer import get_tokenizer
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk

from experiments.common import RESULTS, prepare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmlu-epochs", type=int, default=3)
    parser.add_argument("--gsm8k-epochs", type=int, default=4)
    args = parser.parse_args()
    prepare()
    tokenizer = get_tokenizer()
    sources = [
        ("MMLU auxiliary_train", MMLU(subset="all", split="auxiliary_train"), args.mmlu_epochs),
        ("GSM8K train", GSM8K(subset="main", split="train"), args.gsm8k_epochs),
        ("SmolTalk train", SmolTalk(split="train"), 1),
    ]
    output = {"stages": {}, "assistant_mask_example": {}}
    for name, dataset, repetitions in sources:
        output["stages"][name] = {
            "rows_per_epoch": len(dataset),
            "repetitions_in_training_mixture": repetitions,
            "mixture_rows": len(dataset) * repetitions,
            "first_rows": [dataset[index] for index in range(min(3, len(dataset)))],
        }
    conversation = sources[-1][1][0]
    ids, mask = tokenizer.render_conversation(conversation)
    assistant_example = []
    for index, (token_id, supervised) in enumerate(zip(ids, mask)):
        assistant_example.append({
            "token_index": index,
            "token_id": int(token_id),
            "text": tokenizer.decode([int(token_id)]),
            "loss_target": bool(index > 0 and mask[index]),
        })
    output["assistant_mask_example"] = {
        "conversation": conversation,
        "tokens": assistant_example,
        "prediction_targets_are_shifted_one_position": True,
    }
    path = RESULTS / "task_3_data_inspection.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Wrote {path}")
    for label, values in output["stages"].items():
        print(f"{label}: {values['rows_per_epoch']:,} rows/epoch; {values['mixture_rows']:,} rows in mixture")
        for row in values["first_rows"]:
            print(json.dumps(row, ensure_ascii=False, default=str))
    print("Actual SmolTalk example with tokenizer.render_conversation() mask:")
    for token in assistant_example:
        print(json.dumps(token, ensure_ascii=False))


if __name__ == "__main__":
    main()
