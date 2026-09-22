# Report outline (8 pages max, including figures and references)

The page budget is the real constraint: the marked questions add up to roughly 7.5 pages
of required content before you write a word of your own framing. Write to this skeleton
rather than writing prose and cutting later. Every bullet names the artefact that answers
it, so nothing gets written from memory.

The AI self-disclosure appendix is **not** counted in the 8 pages — fill it in honestly
as you go rather than reconstructing it at the end.

---

## Task 0 — Hub warm-up (~0.25 p, required but unmarked)

Table of `allenai/OLMo-1B`, `HuggingFaceTB/SmolLM2-360M`, `Qwen/Qwen2.5-0.5B`:
training tokens/steps, layers, hidden dim, attention heads, trainable parameters, data
sources. **Note the source of each field** (model card / `config.json` / paper), which is
what is actually being assessed. Keep these three handy — Task 4.3 asks you to place your
own model beside published ones, and you already have two of the right size here.

## Task 1 — Tokenization (~1 p, 15 marks)

| Question | Evidence |
|---|---|
| What is BPE; how it differs from word- and character-level | Your merge tree, `assignment/results/merge_tree.txt` |
| Merge tree for **your own** example, with merge counts | `assignment/bpe_merge_tree.py` (`--dot` gives a clean figure) |
| Vocab size vs compression ratio and sequence length | `assignment/results/tok_compare.md`, both vocab sizes, English sample |
| Why vocab size matters downstream: embedding matrix, softmax denominator, rare tokens | `assignment/model_sizing.py` — at depth 2, vocab 32,768 makes embeddings **97%** of all parameters. Quantify it; do not hand-wave |
| Artefacts and failure cases: numbers, source code, non-English | `tok_compare` probes |

Worth a sentence: nanochat's split pattern uses `\p{N}{1,2}`, not GPT-4's `\p{N}{1,3}`
(see the comment in `nanochat/tokenizer.py`). That is a deliberate choice about how many
tokens to spend on numbers at small vocab sizes, and it shows up in your numbers probe.

## Task 2 — Pre-training (~2 p, 25 marks)

1. **The depth parameter.** `model_dim = ceil(depth x 64 / head_dim) x head_dim`,
   `num_heads = model_dim / head_dim`, in `build_model_meta()`. Report layers, heads,
   embedding dim, total parameters, total training tokens — all printed by
   `assignment/model_sizing.py` and logged to the `"model"` record in your `.jsonl`.
   Single-dial pros: compute-optimal by construction, no hyperparameter search, results
   comparable along one axis. Cons: you cannot vary width independently of depth, and the
   derived rules were tuned at d12 and extrapolated muP-style — at depth 2 you are far
   outside the range where they were fitted. Note `head_dim` defaults to 128, so depth 2
   gives exactly **one** attention head.
2. **Scaling laws.** Chinchilla says ~20 tokens/parameter; nanochat defaults to
   `--target-param-data-ratio=12` (and the speedrun deliberately uses 8). State both,
   compute the Chinchilla-optimal horizon for your parameter count, and say which of the
   two your run is. Careful and worth a sentence: nanochat's ratio counts only
   `transformer_matrices + lm_head`, not embeddings — with 97% of your parameters in
   embeddings, "tokens per parameter" means something very different here than in the
   Chinchilla paper.
3. **Loss analysis.** bpb curves from `assignment/plot_runs.py`. Where it flattens, and
   what the train/val gap says. At this scale you will likely see no meaningful
   overfitting gap — say so and explain why (one epoch over fresh web text; the model has
   nowhere near the capacity to memorise it).
4. **Qualitative inspection.** Five raw completions, no system prompt. `base_train`
   already samples with `--sample-every`; they are in your log.

## Task 3 — Mid-training and SFT (~3 p, 30 marks)

1. **Implementation.** Describe `--stage` / `--source` / `--out-tag` and the new
   `midtrain_checkpoints` directory. Say why a separate directory rather than reusing
   `chatsft_checkpoints`: `find_largest_model()` guesses a model tag by regex on `d<n>`,
   so two tags in one directory can silently load the wrong checkpoint. Note that the
   validation mixture is held fixed across stages so the two bpb curves are comparable.
   Show `git diff upstream/master` as the ground truth. **Flag the discrepancy with the
   assignment text**: current nanochat's `train_tasks` no longer contains identity or
   spelling tasks (those existed in the Oct-2025 version), so "remove all other
   conversational and spelling tasks" reduces to removing SmolTalk.
2. **Loss masking.** Why loss is computed only on assistant tokens. Build the concrete
   token-level example from `tokenizer.render_conversation()`, which returns the mask
   directly — `visualize_tokenization()` prints it for you. Do not invent the example.
3. **Dataset analysis.** Per stage: which data, how many rows, why it belongs to that
   stage. Run the sanity check the assignment asks for — print the first rows of MMLU
   `auxiliary_train` and SmolTalk and show what one training example actually looks like.
   Values/bias: MMLU is US-curriculum multiple choice, SmolTalk is synthetic and
   model-generated, GSM8K is grade-school arithmetic in English. Say what that selects for.
4. **Benchmark scores.** Table of ARC-Easy / ARC-Challenge / GSM8K after **all three**
   stages (base, mid, SFT), from `assignment/results/eval_*.log`. Report the random
   baseline (25% for the multiple-choice tasks) next to your numbers, and be honest: at
   depth 2, expect scores at or near chance. The interesting result is not the score, it
   is that the *base* model cannot even emit a valid answer format while the SFT model
   can. Note the non-determinism caveat the assignment raises.
5. **PEFT.** LoRA/QLoRA vs full fine-tuning: compute, memory, quality. Then the honest
   answer for your case — LoRA saves optimizer memory proportional to trainable
   parameters, but 97% of your parameters are embeddings that LoRA does not usually touch,
   and your memory bottleneck is the logits tensor, not the optimizer. So: no.

## Task 4 — Inference, deployment, reflection (~1.5 p, 30 marks)

**Part A (20)**
1. Autoregressive sampling and the KV cache (`nanochat/engine.py`). Temperature sweep at
   0.1 / 0.7 / 1.5, five examples from your own model, discussed.
2. Three concrete production-hardening changes. Ground them in what you can see in
   `engine.py` / `chat_cli.py` — e.g. no request batching or queueing, no cancellation or
   timeout handling, fixed-size KV cache with no eviction policy. Specific beats generic.
3. Capability comparison: your scores beside two published models (one similar size, one
   much larger) on ARC-E/ARC-C/GSM8K. **Cite every source.** Your Task 0 models cover the
   small end.

**Part B (10)** — pick one prompt and commit to it. Prompt 3 (environmental cost) is the
one where you have your own measurements: `total_training_time_s` from the `"summary"`
record, your GPU's TDP, and the Dutch grid carbon intensity give a defensible number via
[mlco2.github.io](https://mlco2.github.io/impact/). Prompt 1 is also strong if you frame
it around what *you* actually needed to get this running: the cost barrier for you was
not the $100, it was owning a GPU new enough to have Triton support.

---

## Things to have ready for the live grading

Both of you will be asked separately, and can get different grades. Make sure each of you
can, without notes:

- Point at the lines you changed and explain why.
- Explain why bpb rather than loss (vocabulary invariance) — this comes up every time.
- Justify depth, vocab size, batch size, and sequence length as *your* decisions with
  *your* reasons, not as defaults you inherited.
- Say what you would do with 10x the compute, and what you expect would change.
