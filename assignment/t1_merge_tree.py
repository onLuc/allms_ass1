"""Print the BPE merge tree of one word, with the pair count of every merge at the moment
it was learned. Run from the repo root: python -m assignment.merge_tree"""
import os, regex, functools
import matplotlib.pyplot as plt
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

@functools.cache  # the tree is printed and drawn, so count each merge only once
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

def draw(tokens):
    """Draw the merge tree(s) and save them as a PNG. Leaves are raw bytes, the root is the final token."""
    fig, ax = plt.subplots()
    state = {"x": 0, "depth": 0}  # next free leaf position, deepest level reached
    label = lambda t: t.decode("utf-8", "backslashreplace").replace(" ", "␣")  # show spaces as a visible symbol

    def place(token, depth):
        state["depth"] = max(state["depth"], depth)
        rank = ranks[token]
        if rank < 256:  # raw byte = leaf
            x, text, color = state["x"], label(token), "#e8e8e8"
            state["x"] += 1
        else:
            a, b = bpe(token, rank)
            xs = [place(a, depth + 1), place(b, depth + 1)]
            x = sum(xs) / 2
            for xc in xs:
                ax.plot([x, xc], [-depth, -depth - 1], color="gray", lw=1, zorder=1)
            text, color = f"{label(token)}\n#{rank - 255} | {pair_count(a, b, rank):,}", "#cfe2f3"
        ax.text(x, -depth, text, ha="center", va="center", fontsize=9, zorder=2,
                bbox=dict(boxstyle="round,pad=0.3", fc=color, ec="gray"))
        return x

    for t in tokens:
        place(t, 0)
    ax.set_xlim(-0.7, state["x"] - 0.3)
    ax.set_ylim(-state["depth"] - 0.5, 0.5)
    ax.axis("off")
    ax.set_title(f"BPE merge tree for {WORD!r} (vocab size {VOCAB:,})\nnode: token | merge number | pair count when merged")
    fig.set_size_inches(max(6, 1.1 * state["x"]), 1.3 * (state["depth"] + 1) + 0.8)
    os.makedirs("assignment/results", exist_ok=True)
    path = f"assignment/results/merge_tree_{VOCAB}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"\nSaved tree to {path}")

tokens = bpe(WORD.encode("utf-8"), len(ranks))
print(f"{WORD!r} -> {len(tokens)} token(s): {tokens}\n")
for t in tokens:
    show(t)
draw(tokens)