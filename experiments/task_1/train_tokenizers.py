import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from nanochat.common import get_base_dir
from nanochat.dataset import parquets_iter_batched
from nanochat.tokenizer import RustBPETokenizer


def text_iterator(sample_bytes):
    used = 0
    documents = 0
    text_iterator.sample_metadata = {"utf8_bytes": 0, "documents": 0}
    for batch in parquets_iter_batched(split="train"):
        for document in batch:
            raw = document.encode("utf-8")
            remaining = sample_bytes - used
            if remaining <= 0:
                return
            if len(raw) > remaining:
                raw = raw[:remaining]
                document = raw.decode("utf-8", errors="ignore")
                raw = document.encode("utf-8")
            if raw:
                used += len(raw)
                documents += 1
                yield document
            if used >= sample_bytes:
                text_iterator.sample_metadata = {"utf8_bytes": used, "documents": documents}
                return
    text_iterator.sample_metadata = {"utf8_bytes": used, "documents": documents}


def train(vocab_size, sample_bytes, force):
    base = Path(get_base_dir())
    output = base / f"tokenizer_{vocab_size}"
    if output.exists():
        if not force:
            raise FileExistsError(f"{output} exists; pass --force to replace it")
        shutil.rmtree(output)
    started = time.monotonic()
    tokenizer = RustBPETokenizer.train_from_iterator(text_iterator(sample_bytes), vocab_size)
    seconds = time.monotonic() - started
    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output))
    special = {tokenizer.encode_special(token) for token in tokenizer.get_special_tokens()}
    token_bytes = [0 if index in special else len(tokenizer.decode_single_token_bytes(index)) for index in range(vocab_size)]
    torch.save(torch.tensor(token_bytes, dtype=torch.int32), output / "token_bytes.pt")
    sample = text_iterator.sample_metadata
    record = {
        "vocab_size": vocab_size,
        "sample_utf8_bytes": sample["utf8_bytes"],
        "sample_documents": sample["documents"],
        "training_seconds": seconds,
        "tokenizer_directory": str(output),
    }
    (output / "experiment.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-sizes", nargs="+", type=int, default=[8192, 32768])
    parser.add_argument("--sample-bytes", type=int, default=500_000_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for size in args.vocab_sizes:
        train(size, args.sample_bytes, args.force)


if __name__ == "__main__":
    main()
