#!/bin/bash
# =============================================================================
# Assignment 1 pipeline, tuned for a single consumer GPU (GTX 1070, 8 GB, Pascal).
#
#   bash assignment/run_pipeline.sh smoke    # ~10 min: exercises every stage, tiny
#   bash assignment/run_pipeline.sh tok      # Task 1: two tokenizers + evidence
#   bash assignment/run_pipeline.sh pretrain # Task 2
#   bash assignment/run_pipeline.sh mid      # Task 3 stage 1
#   bash assignment/run_pipeline.sh sft      # Task 3 stage 2
#   bash assignment/run_pipeline.sh evals    # Task 3/4 benchmark table
#   bash assignment/run_pipeline.sh all      # everything except smoke
#
# Run it inside screen/tmux. Every stage is independent and re-runnable.
# =============================================================================
set -euo pipefail

STAGE="${1:-smoke}"

# ---- hardware-specific settings ---------------------------------------------
# DEPTH:  2 is the assignment's safe default; 4 only if you have the hours.
# VOCAB:  the vocabulary you pretrain with. It fixes the embedding/lm_head size AND
#         the size of the logits tensor, which is what actually fills 8 GB of VRAM.
# DBS:    device-batch-size. The binding constraint is the logits tensor:
#         DBS x SEQ x VOCAB x 4 bytes, several times over for autograd.
# SEQ:    max-seq-len. Halving it halves the logits memory too.
DEPTH="${DEPTH:-2}"
VOCAB="${VOCAB:-32768}"
DBS="${DBS:-4}"
SEQ="${SEQ:-1024}"
SHARDS="${SHARDS:-3}"
TAG="${TAG:-d${DEPTH}v${VOCAB}}"

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
export OMP_NUM_THREADS=1

# Pascal (compute capability 6.1) has no Triton support, so inductor cannot compile
# anything. Without this, every script dies at torch.compile().
export TORCHDYNAMO_DISABLE=1

# Do NOT set NANOCHAT_DTYPE=float16 on this card: GP104 runs fp16 arithmetic at 1/64
# the fp32 rate (NVIDIA Pascal Tuning Guide), and the logits are cast to fp32 for the
# loss anyway, so it costs speed without buying back the memory that matters.

mkdir -p "$NANOCHAT_BASE_DIR" assignment/results assignment/figures

# SDPA cannot do sliding-window attention, so use full attention on every layer.
COMMON_TRAIN_FLAGS=(
  --window-pattern=L
  --device-batch-size="$DBS"
  --max-seq-len="$SEQ"
  --core-metric-every=-1      # CORE is 22 tasks; far too slow to run mid-training here
)

banner() { echo; echo "================ $* ================"; echo; }

# -----------------------------------------------------------------------------
smoke() {
  banner "SMOKE TEST: every stage, absurdly small. Fail here, not 3 hours in."
  python -m nanochat.dataset -n 1
  python -m scripts.tok_train --max-chars=20000000 --vocab-size=8192 \
      --out-dir="$NANOCHAT_BASE_DIR/tokenizer_smoke"
  rm -rf "$NANOCHAT_BASE_DIR/tokenizer"
  cp -r "$NANOCHAT_BASE_DIR/tokenizer_smoke" "$NANOCHAT_BASE_DIR/tokenizer"

  python -m scripts.base_train --depth=2 "${COMMON_TRAIN_FLAGS[@]}" \
      --num-iterations=20 --eval-every=10 --eval-tokens=8192 --log-every=1 \
      --sample-every=-1 --model-tag=smoke
  python -m scripts.chat_sft --stage=mid --source=base --model-tag=smoke --out-tag=smoke \
      --num-iterations=10 --eval-every=5 --eval-tokens=8192 --chatcore-every=-1 \
      --device-batch-size="$DBS" --max-seq-len="$SEQ" --log-every=1
  python -m scripts.chat_sft --stage=sft --source=mid --model-tag=smoke --out-tag=smoke \
      --num-iterations=10 --eval-every=5 --eval-tokens=8192 --chatcore-every=-1 \
      --device-batch-size="$DBS" --max-seq-len="$SEQ" --log-every=1
  python -m scripts.chat_eval -i sft -g smoke -a ARC-Easy -x 20 -b 2
  banner "SMOKE TEST PASSED — the pipeline runs end to end on this machine."
}

# -----------------------------------------------------------------------------
tok() {
  banner "TASK 1: tokenizers at 8,192 and 32,768 on a 500 MB sample"
  python -m nanochat.dataset -n "$SHARDS"
  for V in 8192 32768; do
    python -m scripts.tok_train --max-chars=500000000 --vocab-size="$V" \
        --out-dir="$NANOCHAT_BASE_DIR/tokenizer_$V" 2>&1 \
        | tee "assignment/results/tok_train_$V.log"
  done
  python -m assignment.tok_compare \
      --tokenizers "$NANOCHAT_BASE_DIR/tokenizer_8192" "$NANOCHAT_BASE_DIR/tokenizer_32768" \
      --out assignment/results/tok_compare.json \
      | tee assignment/results/tok_compare.md
  python -m assignment.bpe_merge_tree --word " tokenization" \
      --out assignment/results/merge_tree.json --dot assignment/results/merge_tree.dot \
      | tee assignment/results/merge_tree.txt

  # Install the chosen tokenizer as THE tokenizer. After this point it is frozen:
  # retraining it invalidates every checkpoint, because vocab_size is baked into the
  # model config and checkpoint_manager asserts the two match.
  rm -rf "$NANOCHAT_BASE_DIR/tokenizer"
  cp -r "$NANOCHAT_BASE_DIR/tokenizer_$VOCAB" "$NANOCHAT_BASE_DIR/tokenizer"
  python -m scripts.tok_eval | tee assignment/results/tok_eval_$VOCAB.log
  banner "Tokenizer frozen at vocab_size=$VOCAB"
}

# -----------------------------------------------------------------------------
pretrain() {
  banner "TASK 2: pretraining depth=$DEPTH"
  python -m assignment.model_sizing --depths "$DEPTH" --vocab-sizes "$VOCAB" \
      --json-out assignment/results/sizing_$TAG.json | tee assignment/results/sizing_$TAG.txt
  # eval-every/log-every are deliberately small: the compute-optimal run is only a few
  # hundred steps, so nanochat's defaults (250/100) would give a 2-point loss curve.
  python -m scripts.base_train --depth="$DEPTH" "${COMMON_TRAIN_FLAGS[@]}" \
      --model-tag="$TAG" \
      --eval-every=20 --eval-tokens=2097152 --log-every=5 \
      --sample-every=100 --save-every=200 \
      2>&1 | tee "assignment/results/base_train_$TAG.log"
  python -m scripts.base_eval --device-batch-size="$DBS" --max-per-task=200 \
      2>&1 | tee "assignment/results/base_eval_$TAG.log"
}

# -----------------------------------------------------------------------------
mid() {
  banner "TASK 3 stage 1: mid-training (MMLU + GSM8K only)"
  python -m scripts.chat_sft --stage=mid --source=base \
      --model-tag="$TAG" --out-tag="$TAG" \
      --device-batch-size="$DBS" --max-seq-len="$SEQ" \
      --num-iterations=400 --eval-every=25 --eval-tokens=2097152 \
      --chatcore-every=-1 --log-every=5 \
      2>&1 | tee "assignment/results/midtrain_$TAG.log"
  evals_for mid
}

sft() {
  banner "TASK 3 stage 2: SFT (SmolTalk only), starting from the mid-training checkpoint"
  python -m scripts.chat_sft --stage=sft --source=mid \
      --model-tag="$TAG" --out-tag="$TAG" \
      --device-batch-size="$DBS" --max-seq-len="$SEQ" \
      --num-iterations=400 --eval-every=25 --eval-tokens=2097152 \
      --chatcore-every=-1 --log-every=5 \
      2>&1 | tee "assignment/results/sft_$TAG.log"
  evals_for sft
}

# -----------------------------------------------------------------------------
# -x caps problems per task. Raise it for the final numbers; the cap is the single
# biggest lever on how long evaluation takes on this card.
evals_for() {
  local SRC="$1"
  banner "Evaluating source=$SRC tag=$TAG"
  python -m scripts.chat_eval -i "$SRC" -g "$TAG" \
      -a "ARC-Easy|ARC-Challenge|GSM8K" -x "${EVAL_MAX:-300}" -b 4 \
      2>&1 | tee "assignment/results/eval_${SRC}_$TAG.log"
}

evals() {
  # All three stages on the same axis, which is what Task 3 question 4 asks for.
  evals_for base
  evals_for mid
  evals_for sft
  python -m assignment.plot_runs "$NANOCHAT_BASE_DIR"/logs/*.jsonl --out assignment/figures
}

case "$STAGE" in
  smoke) smoke ;;
  tok) tok ;;
  pretrain) pretrain ;;
  mid) mid ;;
  sft) sft ;;
  evals) evals ;;
  all) tok; pretrain; mid; sft; evals ;;
  *) echo "unknown stage: $STAGE"; exit 1 ;;
esac
