"""Print the BPE merge tree of one word, with the pair count of every merge at the moment
it was learned. Run from the repo root: python -m assignment.t1_merge_tree"""


import os, regex
from collections import Counter
from nanochat.tokenizer import RustBPETokenizer
from nanochat.common import get_base_dir
from nanochat.dataset import parquets_iter_batched

WORD = " verification"         # leading space: most word tokens include the space before them
VOCAB = 32768               # which of your two tokenizers to inspect
SAMPLE_BYTES = 100_000_000  # text used to count pairs (500_000_000 = the exact training sample)

enc = RustBPETokenizer.from_directory(os.path.join(get_base_dir(), f"tokenizer_{VOCAB}")).enc
ranks = enc._mergeable_ranks  # token bytes -> rank; ranks 0-255 are raw bytes, 256+ are merges in learned order

def bpe(b, max_rank):
    """Split bytes b into tokens using only merges learned before max_rank."""
    parts = [bytes([x]) for x in b]
    while len(parts) > 1:
        r, i = min((ranks.get(parts[i] + parts[i + 1], max_rank), i) for i in range(len(parts) - 1))
        if r >= max_rank:
            break
        parts[i:i + 2] = [parts[i] + parts[i + 1]]
    return parts

# Count pre-tokenized chunks the same way the trainer does (same split regex, same data order)
chunks, nbytes = Counter(), 0
for batch in parquets_iter_batched(split="train"):
    for doc in batch:
        chunks.update(c.encode("utf-8") for c in regex.findall(enc._pat_str, doc))
        nbytes += len(doc.encode("utf-8"))
    if nbytes >= SAMPLE_BYTES:
        break

def pair_count(a, b, rank):
    """How often the pair (a, b) occurred when merge `rank` was learned."""
    total = 0
    for chunk, freq in chunks.items():
        if a + b in chunk:
            p = bpe(chunk, rank)
            total += freq * sum(p[i] == a and p[i + 1] == b for i in range(len(p) - 1))
    return total

def show(token, indent=""):
    rank = ranks[token]
    if rank < 256:
        print(f"{indent}{token!r}  (raw byte)")
        return
    a, b = bpe(token, rank)  # the two tokens that were merged to create this one
    print(f"{indent}{token.decode('utf-8', 'replace')!r}  merge #{rank - 255}, count {pair_count(a, b, rank):,}")
    show(a, indent + "    ")
    show(b, indent + "    ")

tokens = bpe(WORD.encode("utf-8"), len(ranks))
print(f"{WORD!r} -> {len(tokens)} token(s): {tokens}\n")
for t in tokens:
    show(t)
