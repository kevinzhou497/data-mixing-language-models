# 1B Model Experiment Handoff

## Overview

The 1B Wikipedia experiments mirror what was done for 30M/124M/345M/757M. Each run trains a 1B GPT on a mix of FineWeb (primary) and Wikipedia (secondary), sweeping `--mixing_ratio` (= fraction of each batch drawn from FineWeb) to find the minimum Wikipedia validation loss at each training horizon.

There are two experiment tracks:

1. **Scaling Laws** — subsample=1 (full Wikipedia), 5 training horizons doubling in length
2. **Repeat-Aware** — fixed target horizon (114,142 iters), with subsampled Wikipedia (1/2, 1/4, 1/8, 1/16 of docs) and proportionally fewer iterations, preserving the number of repetitions

---

## Submitting Jobs

```bash
cd $HOME/data-mixing-language-models
qsub job_1b.sh           # edit MIX_RATIOS / ITERATIONS before submitting
qsub job_1b_second.sh    # same
```

Check status: `qstat -u $USER`. Stdout/stderr go to `job_o/` and `job_e/`.

The 1B model requires **2 GPUs** (`ngpus=2`, `torchrun --nproc_per_node=2`) and `--grad_checkpointing`. Do not remove the checkpointing flag — it's needed to fit in memory.

---

## Mixing Ratio Search Strategy

- Always fix `LEARNING_RATES=(0.001)` and `SAMPLE` as appropriate for the track.
- Sweep `MIX_RATIOS` in **0.05 increments**.
- After a batch completes, read the summary log. Keep searching upward if val loss is still decreasing; stop when it turns back up. The lowest val loss point is the optimal.
- Start each new horizon at the lower end of the expected range (see tables below) so you establish the trend before committing more compute.

Results append to:
```
summary_logs/wikipedia/summary_<ITERS>iter_1B_<SAMPLE>.txt
```

Per-run output files (all inside `logs/wikipedia/<ITERS>iters/mix<MIX>_lr0.001_1B_<SAMPLE>/`):

| File | Contents |
|---|---|
| `train.log` | Stdout/stderr from the torchrun process |
| `<run_id>.txt` | Step-by-step train loss and val loss (text) |
| `train_losses.npy` | Per-batch train loss for every training step, shape `[num_iterations]` |
| `val_seq_losses_step{N:06d}.npy` | Per-sequence val losses at step N, shape `[val_seqs]`; saved every 100 steps |
| `val_seq_losses_final.npy` | Per-sequence val losses at the final step — **use this for bootstrap CIs** |
| `val_seq_losses_2_step{N:06d}.npy` | Same for second val bin (if `--val_bin_2` passed) |
| `val_seq_losses_2_final.npy` | Same for second val bin, final step |
| `state_step{N:06d}.pt` | Model checkpoint (weights only) at step N |

The `val_seq_losses_final.npy` file is the key artifact for bootstrap confidence interval analysis — each entry is the mean cross-entropy loss for one validation sequence (~512 sequences total). Resample across runs at different mixing ratios to get error bars on the optimal mixing ratio.

---

## Track 1: Scaling Laws (subsample=1)

Full Wikipedia, 5 horizons. Edit `job_1b.sh` or `job_1b_second.sh` with the appropriate `ITERATIONS` and `MIX_RATIOS`.

| Horizon | Iterations | ~Tokens | Status | Tried So Far |
|---|---|---|---|---|
| 1 | 7,134 | 234M | **Re-running** (need val seq losses) | 0.00, 0.05, 0.10 → optimal **0.00** (confirmed; re-running in `job_1b_second.sh`) |
| 2 | 14,268 | 467M | In progress | Done: 0.05–0.15; Running: 0.20–0.30 |
| 3 | 28,536 | 935M | Not started | Was running 0.25–0.35 but those runs predate val seq loss saving; needs re-run |
| 4 | 57,071 | 1.87B | Not started | — |
| 5 | 114,142 | 3.74B | Not started | — |

**Expected optimal ranges** (based on 757M results — 1B may shift slightly):

| Horizon | Iters | 757M Optimal | Expected 1B Range |
|---|---|---|---|
| 1 | 7,134 | 0.10 | **0.00** (confirmed) |
| 2 | 14,268 | 0.40 | 0.25–0.45 |
| 3 | 28,536 | 0.65 | 0.45–0.70 |
| 4 | 57,071 | 0.75 | 0.65–0.80 |
| 5 | 114,142 | 0.85 | 0.75–0.90 |

**Ordered batches still needed:**

*Horizon 2 (14,268 iters, SAMPLE=1):*
- Batch 2: 0.20, 0.25, 0.30 ← currently in `job_1b.sh`
- Batch 3 (if still decreasing): 0.35, 0.40, 0.45
- Batch 4 (if still decreasing): 0.50

*Horizon 3 (28,536 iters, SAMPLE=1):*
- Batch 1: 0.25, 0.30, 0.35 ← needs re-run (previous runs predate val seq loss saving)
- Batch 2 (if still decreasing): 0.40, 0.45, 0.50
- Batch 3 (if still decreasing): 0.55, 0.60, 0.65
- Batch 4 (if still decreasing): 0.70

*Horizon 4 (57,071 iters, SAMPLE=1):*
- Batch 1: 0.55, 0.60, 0.65 ← start here
- Batch 2 (if still decreasing): 0.70, 0.75
- Batch 3 (if still decreasing): 0.80

*Horizon 5 (114,142 iters, SAMPLE=1):*
- Batch 1: 0.70, 0.75, 0.80 ← start here
- Batch 2 (if still decreasing): 0.85, 0.90

---

## Track 2: Repeat-Aware (subsampled Wikipedia)

These experiments use progressively smaller subsets of Wikipedia documents with proportionally fewer training iterations, preserving the number of repetitions relative to the target horizon (114,142 iters with SAMPLE=1). The iteration counts differ slightly from the scaling laws horizons because they are derived from subsampled dataset sizes.

Use `data/wikitext/subsample_<S>_docs/train_*.bin` for the secondary source and set `SAMPLE=<S>` in the job script. Validation always uses `data/wikitext/subsample_1_docs/validation_*.bin`.

| Subsample | Iters | ~Tokens | Status | Notes |
|---|---|---|---|---|
| 1/2 (SAMPLE=2) | 57,118 | ~1.87B | Not started | Sweep from ~0.55–0.80 |
| 1/4 (SAMPLE=4) | 28,566 | ~935M | Not started | Sweep from ~0.45–0.70 |
| 1/8 (SAMPLE=8) | 14,280 | ~467M | Not started | Sweep from ~0.25–0.50 |
| 1/16 (SAMPLE=16) | 7,110 | ~234M | Not started | Sweep from ~0.05–0.20 |

Iteration counts are confirmed from the existing 30M/124M/345M/757M summary logs in `summary_logs/wikipedia/`.

To run repeat-aware jobs, copy `job_1b.sh` to a new file (e.g., `job_1b_ra_s8.sh`) and update:
```bash
ITERATIONS=(14280)   # adjust to match subsampled dataset size
SAMPLE=8             # sets which subsample directory is used
MIX_RATIOS=(...)     # start with expected range
```

---

## Interpreting Results

The summary log appends one line per completed run:
```
mixing_ratio=0.40, learning_rate=0.001, val_loss=3.1590, perplexity=23.54
```

Sort by val_loss to find the minimum. If the minimum is at the edge of the range you tried, keep searching in that direction. A typical sweep looks like:

```
mix=0.30 → 3.18   (still decreasing)
mix=0.35 → 3.16
mix=0.40 → 3.159  ← minimum
mix=0.45 → 3.172  (rising → done)
```

---

## PubMed Extension

Same process as Wikipedia. Key differences in the job script:

```bash
HQ_DATASET="pubmed"
# train secondary:
--train_bin_secondary "data/pubmed/train_subsamples/subsample_${SAMPLE}_docs/train_*.bin"
# validation:
--val_bin "data/pubmed/val_200K/validation_*.bin"
```

Results go to `summary_logs/pubmed/summary_<ITERS>iter_1B_<SAMPLE>.txt`.

### Scaling Laws (SAMPLE=1)

| Horizon | Iters | 757M Optimal | Expected 1B Range |
|---|---|---|---|
| 1 | 7,324 | 0.15 | 0.05–0.20 |
| 2 | 14,649 | 0.40 | 0.25–0.50 |
| 3 | 29,297 | 0.60 | 0.45–0.70 |
| 4 | 58,594 | 0.75 | 0.60–0.85 |
| 5 | 117,188 | 0.80 | 0.70–0.90 |

### Repeat-Aware

| Subsample | Iters |
|---|---|
| 1/2 (SAMPLE=2) | 58,566 |
| 1/4 (SAMPLE=4) | 29,249 |
| 1/8 (SAMPLE=8) | 14,679 |
| 1/16 (SAMPLE=16) | 7,366 |
