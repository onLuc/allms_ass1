# Environment setup

Do this **first**, before any training. The two hardware blockers below are the most
likely reason to lose a day on this assignment, and both are visible within 15 minutes.

The repo is pinned to upstream commit `92d63d4` (2026-07-03). Record that in the report:
nanochat changes weekly, and the assignment PDF was written against an older version.

```bash
git clone https://github.com/<you>/nanochat.git && cd nanochat
git remote add upstream https://github.com/karpathy/nanochat.git
git fetch upstream && git checkout -b assignment 92d63d4
```

---

## 1. Install

```bash
uv sync --extra gpu        # CUDA
# uv sync --extra cpu      # CPU / Apple Silicon
source .venv/bin/activate
```

## 2. The 15-minute hardware check

```bash
python - <<'PY'
import torch
print("torch        :", torch.__version__)
print("cuda avail   :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device       :", torch.cuda.get_device_name(0))
    print("capability   :", torch.cuda.get_device_capability(0))
    print("arch list    :", torch.get_arch_list())
    x = torch.randn(1024, 1024, device="cuda")
    print("matmul works :", (x @ x).sum().item() is not None)
PY
```

Read the output against this table.

| What you see | Meaning | Action |
|---|---|---|
| `capability (6, 1)` and `sm_61` **missing** from arch list | Your torch build has no kernels for this GPU. Every CUDA op fails with *"no kernel image is available for execution on the device"* | See **Blocker A** |
| `capability` major < 7 | Triton, and therefore `torch.compile`, cannot target this GPU | See **Blocker B** |
| `capability (8, x)` or higher | You are fine; ignore both blockers | — |

### Blocker A — cu128 wheels dropped Pascal

`pyproject.toml` pulls torch from the `pytorch-cu128` index. CUDA 12.8 builds ship
kernels for `sm_70` and newer only; CUDA 12.6 builds still include `sm_50`/`sm_60`/`sm_61`.
A GTX 1070 is `sm_61`, so the pinned build cannot execute a single kernel on it.

Fix, in order of preference:

```bash
# 1) same version, older CUDA build
uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126

# 2) if 2.9.1 is not published for cu126, take the newest cu126 build that is,
#    and relax the pin in pyproject.toml to match
uv pip install "torch<2.9" --index-url https://download.pytorch.org/whl/cu126
```

Then re-run the hardware check and confirm `sm_61` appears in the arch list. If you had
to drop below torch 2.9, run the smoke test immediately — older torch may trip over
newer API use in the repo, and you want to know that now, not tonight.

If neither works, the GTX 1070 is out and the run moves to LIACS. Decide this on day 1.

### Blocker B — no Triton on Pascal

`base_train.py` and `chat_sft.py` both call `torch.compile(model, dynamic=False)`.
Inductor compiles through Triton, which requires compute capability ≥ 7.0 and refuses
anything older with *"Found NVIDIA GeForce GTX 1070 which is too old to be supported by
the triton GPU compiler"*.

Fix — no code change needed, `torch.compile` becomes a pass-through:

```bash
export TORCHDYNAMO_DISABLE=1
```

`assignment/run_pipeline.sh` sets this for you. Cost: roughly a 1.5–2x slowdown versus a
compiled run. Mention it in the report when you discuss throughput, because it is part of
why your MFU will look poor.

## 3. Other consequences of this GPU, for the record

| Thing | Status on GTX 1070 | What to do |
|---|---|---|
| Flash Attention 3 | Unavailable (Hopper/Ampere kernels only). `flash_attention.py` falls back to SDPA automatically | Nothing — but pass `--window-pattern=L`, because SDPA cannot do sliding-window attention and the run will crawl otherwise |
| bfloat16 | Unavailable below `sm_80`; nanochat auto-selects fp32 | Nothing |
| float16 | GP104 runs fp16 at **1/64** the fp32 rate ([NVIDIA Pascal Tuning Guide](https://docs.nvidia.com/cuda/pascal-tuning-guide/index.html)) | Do **not** set `NANOCHAT_DTYPE=float16`. It is slower, and the logits are cast to fp32 for the loss anyway so it barely saves memory |
| FP8 | H100 only | Never pass `--fp8` |
| VRAM | 8 GB | See below — the logits tensor, not the model, is what you run out of |

### Why you will OOM, and the actual fix

At depth 2 the model is ~13 M parameters: nothing. The memory goes to the logits in
`gpt.py`, which are materialised in fp32 for the whole micro-batch:

```
device_batch_size x max_seq_len x vocab_size x 4 bytes
```

With nanochat's defaults (32 x 2048 x 32768) that is **8.6 GB for one tensor**, before
autograd keeps a second copy for the softcap. Hence `--device-batch-size=4
--max-seq-len=1024` in the run script: 4 x 1024 x 32768 x 4 B = 537 MB, which fits.

Gradient accumulation keeps the *total* batch size unchanged, so this costs wall-clock,
not model quality.

## 4. LIACS fallback

The DSLab machines are shared and may be busy. Keep both paths alive:

- Same commands; drop `TORCHDYNAMO_DISABLE`, raise `--device-batch-size`, and remove
  `--window-pattern=L` if the GPU is Ampere or newer.
- Set `NANOCHAT_BASE_DIR` to somewhere with tens of GB free — checkpoints, data shards
  and HF dataset caches add up fast, and home quotas are small.
- Always run under `screen -L -Logfile run.log -S nanochat`, so a dropped SSH session
  does not kill a two-hour run.
