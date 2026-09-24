"""Train two BPE tokenizers (8,192 and 32,768 vocab) on a 500 MB sample of CLIMBMix
using nanochat's Rust BPE trainer. Run from the repo root: python -m assignment.train_tokenizers"""

"""
How to run:
Make sure all packages are installed using the following commands
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# restart terminal
uv sync --extra gpu
.venv\Scripts\activate
python -m nanochat.dataset -n 8

And then, to resolve nanochat package dependencies, run from root:
python -m assignment.t1_tokenization
"""

import os, time, torch
from nanochat.tokenizer import RustBPETokenizer
from nanochat.common import get_base_dir
from nanochat.dataset import parquets_iter_batched

MAX_BYTES = 500_000_000  # 500 MB of UTF-8 text

def text_iterator():
    nbytes = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            yield doc
            nbytes += len(doc.encode("utf-8"))
            if nbytes >= MAX_BYTES:
                return

for vocab_size in [8192, 32768]:
    t0 = time.time()
    tokenizer = RustBPETokenizer.train_from_iterator(text_iterator(), vocab_size)
    print(f"vocab {vocab_size}: trained in {time.time() - t0:.1f}s")

    out_dir = os.path.join(get_base_dir(), f"tokenizer_{vocab_size}")
    tokenizer.save(out_dir)

    # bytes per token id (0 for special tokens), used by nanochat for bits-per-byte eval
    special = {tokenizer.encode_special(s) for s in tokenizer.get_special_tokens()}
    token_bytes = [0 if i in special else len(tokenizer.decode_single_token_bytes(i)) for i in range(vocab_size)]
    torch.save(torch.tensor(token_bytes, dtype=torch.int32), os.path.join(out_dir, "token_bytes.pt"))