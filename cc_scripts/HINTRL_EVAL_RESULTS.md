# Heldout Eval Results — Hint RL, QuestA, Base Models

**Last updated**: 2026-04-16
**Models evaluated**: Hint RL, QuestA, OpenMath-Nemotron-1.5B, DeepScaleR-1.5B
**Eval settings**: n_samples=32, temperature=0.7, top_p=0.95, max_new_tokens=12000, 1× H100
**Plot**: [heldout_eval_bar.png](plots/figures/heldout_eval_bar.png)

---

## Results Summary

### Pass@1

| Dataset | Hint RL | QuestA | Nemotron (base) | DeepScaleR (base) |
|---------|:-------:|:------:|:---------------:|:-----------------:|
| AIME 2024 (30) | **0.502** | 0.489 | 0.496 | 0.371 |
| AIME 2025 (30) | **0.433** | 0.294 | 0.401 | 0.308 |
| OlympiadBench (1517 / 819†) | **0.332** | 0.173† | 0.332 | 0.305 |
| HMMT Feb 2025 (30) | 0.188 | **0.199** | 0.177 | 0.125 |

### Pass@8

| Dataset | Hint RL | QuestA | Nemotron | DeepScaleR |
|---------|:-------:|:------:|:--------:|:----------:|
| AIME 2024 | **0.756** | 0.706 | 0.746 | 0.625 |
| AIME 2025 | **0.653** | 0.491 | 0.617 | 0.439 |
| OlympiadBench | **0.432** | 0.265† | 0.429 | 0.415 |
| HMMT Feb 2025 | 0.428 | **0.448** | 0.402 | 0.250 |

### Pass@32

| Dataset | Hint RL | QuestA | Nemotron | DeepScaleR |
|---------|:-------:|:------:|:--------:|:----------:|
| AIME 2024 | **0.800** | 0.733 | 0.800 | 0.700 |
| AIME 2025 | **0.733** | 0.567 | 0.733 | 0.567 |
| OlympiadBench | **0.477** | 0.316† | 0.477 | 0.472 |
| HMMT Feb 2025 | 0.533 | **0.567** | 0.500 | 0.400 |

† **QuestA OlympiadBench is partial** — 819/1517 problems (54%) completed before 12h SLURM timeout.
All 819 had full 32 samples; numbers are computed on that subset.

---

## Findings

### 1. HintRL is the strongest on AIME and ties on OlympiadBench
HintRL dominates AIME 2024, AIME 2025, and matches Nemotron base on OlympiadBench.
The biggest gain is on AIME 2025 (+3.2% pass@1 over base, +12% vs DeepScaleR).

### 2. QuestA surprisingly underperforms on most benchmarks
Despite being a well-regarded public baseline, `foreverlasting1202/QuestA-Nemotron-1.5B`
shows lower pass@1 than even the Nemotron base on AIME24/25 and OlympiadBench. It does
win on HMMT (pass@1, 8, 32) — suggesting its hint-based curriculum may overfit to
certain problem styles.

Possible causes (not yet investigated):
- Checkpoint uploaded to HF may not be the best-performing one
- Different sampling parameters may be optimal for QuestA
- QuestA generates much longer chains of thought (which caused the OOM issues below)

### 3. HintRL and Nemotron have similar pass@32 (coverage)
Both solve 80% of AIME24, 73.3% of AIME25, and 47.7% of OlympiadBench at pass@32.
This suggests HintRL's improvement is in **consistency of correct reasoning**, not in
expanding the set of solvable problems.

### 4. HMMT Feb 2025 is the hardest benchmark
All models cluster at pass@1 ≈ 0.12–0.20. Even pass@32 tops out at ~0.57 (QuestA).
This is a useful signal benchmark because all models still have significant headroom.

---

## Data locations

### Full result JSONL files (scratch — may not be preserved)
```
/scratch/fengdic/hint_rl_results/logs/fengdic/eval_{dataset}/eval_{dataset}-{model_tag}/rollout/0/*.jsonl
```

model_tag ∈ `{nemotron_base, deepscaler_base, hintrl, questa}`
dataset ∈ `{aime24, aime25, olympiad_bench, hmmt_feb_2025}`

### Sampled results in repo (survives migration)
```
cc_scripts/results/eval_heldout/eval_{dataset}-{model_tag}-sample.jsonl  (5 problems × 32 samples)
cc_scripts/results/eval_heldout/eval_{dataset}-{model_tag}-config.yaml   (full run config)
```

---

## Reproducing

### 1. Pull models
```bash
source .venv/bin/activate
# Our trained checkpoint (requires HF_TOKEN in .env scoped to FengdiFlo/HintRL)
bash cc_scripts/pull_from_hf.sh FengdiFlo/HintRL /home/fengdic/scratch/models/HintRL

# QuestA public baseline (no token needed)
huggingface-cli download foreverlasting1202/QuestA-Nemotron-1.5B \
    --local-dir /home/fengdic/scratch/models/QuestA-Nemotron-1.5B

# Base models
bash cc_scripts/pull_from_hf.sh nvidia/OpenMath-Nemotron-1.5B \
    /home/fengdic/scratch/models/OpenMath-Nemotron-1.5B
```

### 2. Submit eval jobs (staggered — see gotchas below)
```bash
DAT_FILE=cc_scripts/slurm/eval_heldout-hintrl.dat \
    bash cc_scripts/slurm/resubmit_heldout_eval.sh --no-cancel
# Same pattern for eval_heldout-questa.dat
```

### 3. Generate plot
```bash
python cc_scripts/plots/plot_heldout_bar.py \
    --results-root /scratch/fengdic/hint_rl_results/logs/fengdic \
    --models hintrl questa nemotron_base deepscaler_base \
    --k-values 1 8 32 \
    --output cc_scripts/plots/figures/heldout_eval_bar.png
```

---

## Known Issues & Gotchas

### A. OlympiadBench doesn't fit in 12h with n_samples=32
The 1517-problem set at 32 samples × 12000 max_new_tokens exceeds the 12h SLURM limit
on Rorqual. For QuestA (which generates longer chains), 54% completed.

**Options**:
- Split into 2-3 dat file batches by problem range (requires eval_math.py patch to
  accept `--problem_range` or similar)
- Reduce `n_samples` to 16 (halves runtime, still fine for pass@1, pass@8, pass@16)
- Reduce `gconfig.max_new_tokens` (risks truncating long reasoning)
- Request longer time on cluster with 24h+ partition if available

### B. Port 36355 conflict when multiple eval jobs share a node
Rorqual's 4-GPU nodes can host 4 separate `--gres=gpu:h100:1` jobs. All eval workers
use hardcoded port 36355 → only the first succeeds.

**Fix**: Submit jobs staggered — wait for each to reach RUNNING before submitting next.
See the pattern in `cc_scripts/slurm/eval_heldout_math.sh` (has `wait_for_job_to_start`
helper) — verify this is used in any resubmission script.

### C. SGLang OOM with `mem_fraction_static: 0.9` on QuestA
QuestA generates very long chains of thought (more than HintRL/Nemotron). With
`mem_fraction_static: 0.9`, only ~1 GiB free after KV cache pre-allocation, but
intermediate attention allocations need ~12 GiB.

**Fix applied**: Changed `cc_scripts/configs/eval/eval_math.yaml` to
`mem_fraction_static: 0.75` + `max_running_requests: 32`.
This gives ~17 GiB free, fits all models tested so far.
These are SGLang server-side settings — do NOT affect sampling/generation quality.

### D. SGLang GLIBC bug on Rorqual (patched)
SGLang's `get_nvgpu_memory_capacity()` subprocess spawns `nvidia-smi`, which crashes
due to GLIBC_ABI_DT_RELR on Rorqual compute nodes. Monkey-patched in
`areal/engine/sglang_remote.py`. Re-apply after SGLang upgrades.

### E. OlympiadBench loader null answers
Dataset has ~609 entries with null answers. Filter applied in
`areal/dataset/heldout_math.py:get_olympiad_bench_rl_dataset()`. Problem count is
1517 after filtering.

---

## For migrating to another cluster — critical path

1. **Verify GPU / driver / SGLang combo**: On new cluster, run a smoke test first:
   ```bash
   N_SAMPLES=2 MAX_CONCURRENT=4 bash cc_scripts/slurm/salloc_eval.sh aime24 \
       /path/to/models/OpenMath-Nemotron-1.5B
   ```
   If you get a `nvidia-smi` / GLIBC error, the patch in `areal/engine/sglang_remote.py`
   should already handle it. If GPU isn't detected at all, check whether `salloc` on the
   new cluster keeps your shell on the login node (Rorqual does this — need `srun` or
   auto-dispatch inside `salloc_eval.sh`).

2. **Update paths**: Replace `/home/fengdic/evan_workspace/hint_rl` and
   `/home/fengdic/scratch` with the new cluster paths. Audit:
   - `cc_scripts/configs/eval/eval_math.yaml` (cluster.fileroot, paths in comments)
   - `cc_scripts/slurm/eval_heldout_math.sh` (REPO_ROOT, SCRATCH, ACCOUNT)
   - `cc_scripts/slurm/resubmit_heldout_eval.sh` (same)
   - `cc_scripts/slurm/eval_heldout-*.dat` (actor_path is absolute)

3. **Update SLURM account / GPU type**: Rorqual uses `def-ashique` and `h100`. On the
   new cluster, update `ACCOUNT` and `GPU_TYPE` defaults in the slurm scripts.

4. **Re-pull models to new cluster's scratch** (compute nodes have no internet on
   Rorqual; check policy on new cluster):
   ```bash
   bash cc_scripts/pull_from_hf.sh FengdiFlo/HintRL <new_scratch>/models/HintRL
   huggingface-cli download foreverlasting1202/QuestA-Nemotron-1.5B \
       --local-dir <new_scratch>/models/QuestA-Nemotron-1.5B
   bash cc_scripts/pull_from_hf.sh nvidia/OpenMath-Nemotron-1.5B \
       <new_scratch>/models/OpenMath-Nemotron-1.5B
   bash cc_scripts/pull_from_hf.sh agentica-org/DeepScaleR-1.5B-Preview \
       <new_scratch>/models/DeepScaleR-1.5B-Preview
   ```

5. **Baseline smoke test before any training**:
   Run `cc_scripts/slurm/eval_heldout-base_model.dat` and verify pass@1 matches:
   - Nemotron AIME24: 0.496, Nemotron AIME25: 0.401
   - DeepScaleR AIME24: 0.371, DeepScaleR AIME25: 0.308
   If these match within ~2%, the eval pipeline is correctly ported.

6. **Re-running QuestA OlympiadBench to completion**:
   Since it timed out at 819/1517, the recommended approach is split the `olympiad_bench`
   dataset loader into 2 halves (`olympiad_bench_part1`, `olympiad_bench_part2`) and
   submit as 2 separate jobs. Alternative: reduce `n_samples` to 16.

---

## Outstanding tasks

- [ ] Complete OlympiadBench QuestA eval (currently 54%)
- [ ] Investigate why QuestA underperforms Nemotron base — check if there's a better
      public checkpoint, or if sampling parameters need tuning
- [ ] Generate training dataset `openr1_hint_sep` (required for future HintRL retraining)
- [ ] Generate OpenCode dataset (for code-domain experiments)
- [ ] Adapt training SLURM scripts for target cluster
- [ ] Run hyperparam sweep on HintRL Goldilocks zone
