"""
Task 1, question 1: a BPE merge tree for a concrete example, with the merge counts.

nanochat trains its tokenizer in Rust (rustbpe) and only hands back the final
mergeable_ranks, so the pair *counts* that drove each merge are not recoverable from a
trained tokenizer. This script therefore trains a small BPE from scratch in readable
Python, on real ClimbMix text and with nanochat's exact split pattern, and records the
count of every merge as it happens. The resulting tree is your own example, built from
your own data, which is what the question asks for.

It is not a replacement for scripts/tok_train.py -- it is a microscope for one word.

Run:
    python -m assignment.bpe_merge_tree --word " tokenization" --num-merges 2000
    python -m assignment.bpe_merge_tree --word " Leiden" --dot assignment/results/merges.dot
"""

import argparse
import json
import os
from collections import Counter

import regex as re

from nanochat.tokenizer import SPLIT_PATTERN
from nanochat.dataset import parquets_iter_batched


def load_corpus(max_chars):
    chunks, total = [], 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            chunks.append(doc)
            total += len(doc)
            if total >= max_chars:
                return "".join(chunks)[:max_chars]
    return "".join(chunks)[:max_chars]


def train_bpe(corpus, num_merges):
    """
    Textbook BPE over a word-frequency table.

    Words are byte tuples, so we start from the 256 byte values exactly like nanochat
    does (byte-level BPE never needs an <unk> token). Each round we count adjacent
    pairs weighted by word frequency, merge the most frequent pair, and record it.
    """
    pat = re.compile(SPLIT_PATTERN)
    word_freqs = Counter(pat.findall(corpus))
    # each word becomes a tuple of its UTF-8 bytes, represented as ints 0..255
    words = {tuple(w.encode("utf-8")): f for w, f in word_freqs.items()}

    vocab = {i: bytes([i]) for i in range(256)}
    merges = []  # list of dicts: rank, pair, count, new_id, new_token_bytes

    for rank in range(num_merges):
        pair_counts = Counter()
        for word, freq in words.items():
            for a, b in zip(word, word[1:]):
                pair_counts[(a, b)] += freq
        if not pair_counts:
            break
        (a, b), count = pair_counts.most_common(1)[0]
        if count < 2:
            break
        new_id = 256 + rank
        vocab[new_id] = vocab[a] + vocab[b]
        merges.append({
            "rank": rank,
            "pair": [a, b],
            "pair_repr": [vocab[a].decode("utf-8", "replace"), vocab[b].decode("utf-8", "replace")],
            "count": count,
            "new_id": new_id,
            "new_token": vocab[new_id].decode("utf-8", "replace"),
        })
        # apply the merge everywhere
        new_words = {}
        for word, freq in words.items():
            out, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            new_words[tuple(out)] = new_words.get(tuple(out), 0) + freq
        words = new_words

    return vocab, merges


def encode_word(word, vocab, merges):
    """Re-run the merges on one word, recording which ones fire and in what order."""
    rank_of = {tuple(m["pair"]): m for m in merges}
    seq = list(word.encode("utf-8"))
    history = [list(seq)]
    applied = []
    while len(seq) > 1:
        # the merge with the lowest rank (learned earliest) wins, as in real BPE
        candidates = [(rank_of[(a, b)]["rank"], i, (a, b))
                      for i, (a, b) in enumerate(zip(seq, seq[1:])) if (a, b) in rank_of]
        if not candidates:
            break
        _, i, pair = min(candidates)
        m = rank_of[pair]
        seq = seq[:i] + [m["new_id"]] + seq[i + 2:]
        applied.append(m)
        history.append(list(seq))
    return seq, applied, history


def render_tree(applied, vocab):
    """ASCII merge tree: each line is one merge, deepest (earliest) first."""
    lines = []
    for m in applied:
        left, right = m["pair_repr"]
        lines.append(f"  rank {m['rank']:>5}  count {m['count']:>9,}   "
                     f"{left!r} + {right!r}  ->  {m['new_token']!r}")
    return "\n".join(lines)


def render_dot(word, applied, final_seq, vocab):
    """Graphviz DOT of the merge tree, for a proper figure in the report."""
    lines = ['digraph merges {', '  rankdir=BT;', '  node [shape=box, fontname="monospace"];']
    for m in applied:
        parent = f'n{m["new_id"]}'
        lines.append(f'  {parent} [label="{m["new_token"]}\\nid {m["new_id"]} | count {m["count"]:,}"];')
        for child_id, child_repr in zip(m["pair"], m["pair_repr"]):
            lines.append(f'  n{child_id} [label="{child_repr}"];')
            lines.append(f'  n{child_id} -> {parent};')
    lines.append(f'  label="BPE merge tree for {word!r} -> {len(final_seq)} token(s)";')
    lines.append('}')
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--word", type=str, default=" tokenization", help="the example to build the tree for (leading space matters!)")
    p.add_argument("--num-merges", type=int, default=2000)
    p.add_argument("--corpus-chars", type=int, default=20_000_000)
    p.add_argument("--out", type=str, default="assignment/results/merge_tree.json")
    p.add_argument("--dot", type=str, default=None, help="also write a graphviz .dot file")
    args = p.parse_args()

    print(f"Loading {args.corpus_chars:,} characters of ClimbMix...")
    corpus = load_corpus(args.corpus_chars)
    print(f"Training a {args.num_merges}-merge BPE in pure Python (this is the slow-but-readable one)...")
    vocab, merges = train_bpe(corpus, args.num_merges)
    print(f"Learned {len(merges)} merges. First 10:")
    for m in merges[:10]:
        print(f"  rank {m['rank']:>3}  count {m['count']:>9,}  {m['pair_repr'][0]!r} + {m['pair_repr'][1]!r} -> {m['new_token']!r}")

    final_seq, applied, history = encode_word(args.word, vocab, merges)
    print(f"\nMerge tree for {args.word!r}  ({len(args.word.encode()):,} bytes -> {len(final_seq)} token(s)):\n")
    print(render_tree(applied, vocab))
    print("\nIntermediate states:")
    for state in history:
        print("   " + " | ".join(vocab[t].decode("utf-8", "replace") for t in state))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "word": args.word,
            "corpus_chars": args.corpus_chars,
            "num_merges_learned": len(merges),
            "first_merges": merges[:50],
            "applied_merges": applied,
            "final_token_count": len(final_seq),
        }, f, indent=2)
    print(f"\nwrote {args.out}")

    if args.dot:
        os.makedirs(os.path.dirname(args.dot), exist_ok=True)
        with open(args.dot, "w") as f:
            f.write(render_dot(args.word, applied, final_seq, vocab))
        print(f"wrote {args.dot}  (render with: dot -Tpdf {args.dot} -o merges.pdf)")


if __name__ == "__main__":
    main()
