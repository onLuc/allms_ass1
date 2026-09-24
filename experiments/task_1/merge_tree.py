import argparse
import json
from collections import Counter
from pathlib import Path

import regex
from nanochat.common import get_base_dir
from nanochat.dataset import parquets_iter_batched
from nanochat.tokenizer import RustBPETokenizer

from experiments.common import RESULTS, prepare


def split_bytes(data, max_rank, ranks):
    parts = [bytes([byte]) for byte in data]
    while len(parts) > 1:
        best = max_rank
        at = -1
        for index in range(len(parts) - 1):
            rank = ranks.get(parts[index] + parts[index + 1], max_rank)
            if rank < best:
                best, at = rank, index
        if at < 0 or best >= max_rank:
            break
        parts[at:at + 2] = [parts[at] + parts[at + 1]]
    return parts


def sample_chunks(tokenizer, max_bytes):
    enc = tokenizer.enc
    pattern = regex.compile(enc._pat_str)
    chunks = Counter()
    used = 0
    for batch in parquets_iter_batched(split="train"):
        for document in batch:
            raw = document.encode("utf-8")
            remaining = max_bytes - used
            if remaining <= 0:
                return chunks, used
            if len(raw) > remaining:
                raw = raw[:remaining]
                document = raw.decode("utf-8", errors="ignore")
                raw = document.encode("utf-8")
            used += len(raw)
            chunks.update(piece.encode("utf-8") for piece in pattern.findall(document))
            if used >= max_bytes:
                return chunks, used
    return chunks, used


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--sample-bytes", type=int, default=5_000_000)
    parser.add_argument("--word", default=" counterproductive")
    args = parser.parse_args()
    prepare()
    tokenizer = RustBPETokenizer.from_directory(str(Path(get_base_dir()) / f"tokenizer_{args.vocab_size}"))
    enc = tokenizer.enc
    ranks = enc._mergeable_ranks
    chunks, used = sample_chunks(tokenizer, args.sample_bytes)
    count_cache = {}

    def count_merge(token):
        rank = ranks[token]
        if rank < 256:
            return None
        if token not in count_cache:
            total = 0
            for chunk, frequency in chunks.items():
                parts = split_bytes(chunk, rank, ranks)
                total += frequency * sum(parts[index] + parts[index + 1] == token for index in range(len(parts) - 1))
            count_cache[token] = total
        return count_cache[token]

    serial = 0
    nodes = []
    edges = []

    def build(token, depth=0):
        nonlocal serial
        index = serial
        serial += 1
        rank = ranks[token]
        node = {"id": index, "text": token.decode("utf-8", "replace"), "rank": rank, "merge_count": count_merge(token), "depth": depth}
        nodes.append(node)
        if rank >= 256:
            children = split_bytes(token, rank, ranks)
            if len(children) != 2:
                raise RuntimeError(f"Merge rank {rank} did not resolve to two child tokens")
            for child in children:
                child_id = build(child, depth + 1)
                edges.append((index, child_id))
        return index

    token_ids = tokenizer.encode(args.word)
    pieces = [tokenizer.decode_single_token_bytes(token_id) for token_id in token_ids]
    roots = [build(piece) for piece in pieces]
    lines = [f"Example: {args.word!r}", f"Tokenizer vocab: {args.vocab_size}", f"Merge-count sample: {used:,} UTF-8 bytes from CLIMBMix train"]

    def show(node_id):
        item = nodes[node_id]
        indent = "  " * item["depth"]
        if item["rank"] < 256:
            lines.append(f"{indent}{item['text']!r} (base byte)")
        else:
            lines.append(f"{indent}{item['text']!r} (merge #{item['rank'] - 255}, count {item['merge_count']:,})")
            for parent, child in edges:
                if parent == node_id:
                    show(child)

    for root in roots:
        show(root)
    text_path = RESULTS / "task_1_merge_tree.txt"
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    dot = ["digraph bpe {", "  rankdir=TB;"]
    for node in nodes:
        suffix = f"\ncount={node['merge_count']}" if node["merge_count"] is not None else "\nbase byte"
        dot.append(f"  n{node['id']} [label={json.dumps(node['text'] + suffix, ensure_ascii=False)}];")
    dot.extend(f"  n{parent} -> n{child};" for parent, child in edges)
    dot.append("}")
    dot_path = RESULTS / "task_1_merge_tree.dot"
    dot_path.write_text("\n".join(dot) + "\n", encoding="utf-8")
    (RESULTS / "task_1_merge_tree.json").write_text(json.dumps({"example": args.word, "vocab_size": args.vocab_size, "sample_utf8_bytes": used, "nodes": nodes, "edges": edges}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Wrote {text_path}, {dot_path} and {RESULTS / 'task_1_merge_tree.json'}")


if __name__ == "__main__":
    main()
