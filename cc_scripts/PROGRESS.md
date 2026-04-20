# Hint RL — Progress Log & Migration Guide

**Last updated**: 2026-04-16
**Intended use**: Reference doc for continuing this project on another cluster.
Self-contained; expect to read this + `HINTRL_EVAL_RESULTS.md` + `CORE_KNOWLEDGE.md`
and be able to pick up where things stopped.

---

## TL;DR Current State

| Task | Status | Notes |
|------|--------|-------|
| Codebase understanding | ✅ Done | See `cc_scripts/CORE_KNOWLEDGE.md` |
| HF model sharing set up | ✅ Done | `FengdiFlo/HintRL` + push/pull scripts |
| Base model heldout eval | ✅ Complete | Nemotron + DeepScaleR on 4 datasets |
| HintRL heldout eval | ✅ Complete | One checkpoint, 4 datasets, full 32 samples |
| QuestA heldout eval | ⚠️ Partial | 3 complete + OlympiadBench 54% |
| Comparison plot & report | ✅ Done | [HINTRL_EVAL_RESULTS.md](HINTRL_EVAL_RESULTS.md) |
| Dynamic hint dataset (`openr1_hint_sep`) | ❌ Not generated | Needed to retrain HintRL |
| OpenCode dataset | ❌ Not generated | For code-domain training |
| Training SLURM scripts (Rorqual) | ❌ Not adapted | All hardcoded to chanb/Vulcan |
| HintRL hyperparameter sweep | ❌ Not started | Goldilocks zone exploration |

---

## Key Final Results (pass@1)

| Dataset | HintRL | QuestA | Nemotron base | DeepScaleR base |
|---------|:------:|:------:|:-------------:|:---------------:|
| AIME 2024 | **0.502** | 0.489 | 0.496 | 0.371 |
| AIME 2025 | **0.433** | 0.294 | 0.401 | 0.308 |
| OlympiadBench | **0.332** | 0.173† | 0.332 | 0.305 |
| HMMT Feb 2025 | 0.188 | **0.199** | 0.177 | 0.125 |

† partial (819/1517). See [HINTRL_EVAL_RESULTS.md](HINTRL_EVAL_RESULTS.md) for full tables.

**Key finding**: HintRL matches or beats the Nemotron base on all 4 benchmarks at pass@1,
with the biggest gain on AIME 2025 (+3.2%). QuestA underperforms on AIME/OlympiadBench
despite being a strong baseline on paper.

---

## Environment & Cluster Context (Rorqual)

**Everything below is Rorqual-specific. On migration, the SLURM & path details change.**

- **Cluster**: Rorqual (Alliance Canada)
- **SLURM account**: `def-ashique` (or `rrg-ashique`)
- **GPU**: H100 80GB (`--gres=gpu:h100:N`)
- **Nodes**: 64 CPU cores, 512 GB RAM, 4× H100 per node
- **Modules**: `StdEnv/2023` + `cuda/12.9` (do NOT `module load python` — venv handles it)
- **Python venv**: `/home/fengdic/evan_workspace/hint_rl/.venv`
- **Repo**: `/home/fengdic/evan_workspace/hint_rl`
- **Scratch**: `/home/fengdic/scratch` → `/scratch/fengdic` (20TB quota)
- **Compute nodes have NO internet**: must pre-download HF models & datasets

### Known cluster quirks

1. **`salloc` keeps shell on login node** — must use `srun` to execute on allocated
   GPU node. `cc_scripts/slurm/salloc_eval.sh` does not auto-dispatch today; inside
   salloc do `srun bash cc_scripts/slurm/salloc_eval.sh <dataset>`.
2. **Port 36355 conflict** on same-node multi-job packing. Submit jobs staggered
   (wait for each to RUNNING before next sbatch) to force different nodes.
3. **GLIBC_ABI_DT_RELR bug** in SGLang's `nvidia-smi` subprocess call — patched in
   `areal/engine/sglang_remote.py` via `_patch_sglang_get_device_memory()`.
   Must re-apply after SGLang upgrades; also delete `.pyc` cache.
4. **SLURM script environment (proven working)**:
   ```bash
   module load StdEnv/2023
   module load cuda/12.9
   unset LD_PRELOAD
   source ${REPO_ROOT}/.venv/bin/activate
   export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
   ```

---

## Work Completed — Chronological

### Phase 1 (week of 2026-04-10): Onboarding + infrastructure

- Read `cc_scripts/`, `areal/`, all configs and SLURM scripts.
  Output: [CORE_KNOWLEDGE.md](CORE_KNOWLEDGE.md)
- Set up HuggingFace model sharing: `push_to_hf.sh` + `pull_from_hf.sh`,
  fine-grained token scoped to `FengdiFlo/HintRL` stored in `.env` (gitignored).
- Fixed 3 of 5 heldout dataset loaders that weren't extracting answer fields
  (AIME25, HMMT, BRUMO). Added `get_olympiad_bench_rl_dataset()` loader with
  null-answer filtering.
- Adapted eval SLURM scripts from Vulcan to Rorqual defaults.

### Phase 2 (week of 2026-04-12): Base model heldout eval

- Submitted 8 jobs (2 models × 4 datasets), all with n_samples=32.
- Hit the SGLang GLIBC bug on first runs → patched.
- Completed all 8 runs, full 32 samples. Results in
  [project_heldout_eval_results.md](../.claude/projects/-home-fengdic-evan-workspace-hint-rl/memory/project_heldout_eval_results.md).

### Phase 3 (2026-04-14): HintRL heldout eval

- Pulled checkpoint from `FengdiFlo/HintRL` (3.4 GB safetensors + `hint_percentage.pkl`).
- Created `eval_heldout-hintrl.dat`, submitted 4 jobs.
- All 4 completed on separate nodes (rg31703, rg31803, rg32203, rg31602) — no contention.
- Generated first comparison plot with 3 models.

### Phase 4 (2026-04-15 to 2026-04-16): QuestA heldout eval

- Downloaded `foreverlasting1202/QuestA-Nemotron-1.5B` (2.5 GB, pytorch_model.bin shards).
- **First submission (4 jobs)**: All 4 packed onto 2 nodes → port 36355 conflicts → 3/4 failed.
- **Second submission (3 jobs resubmitted)**: 2 packed onto same node → 2 failures.
- **Root cause found**: CUDA OOM during generation (QuestA generates longer CoT).
  Intermediate attention buffer needs ~12 GiB, but only ~1 GiB free after KV cache
  pre-allocation at `mem_fraction_static: 0.9`.
- **Fix applied**: Changed eval_math.yaml to `mem_fraction_static: 0.75` +
  `max_running_requests: 32`. Provides ~17 GiB free, well above peak allocation.
- **Third submission (staggered, all on separate nodes)**:
  - AIME24, AIME25, HMMT: ✅ completed in ~23 min each
  - OlympiadBench: ⚠️ timed out after 12h with 819/1517 problems complete.
- Saved sampled JSONL + configs to `cc_scripts/results/eval_heldout/` for migration.

---

## Critical Files in the Repo (what you need to know)

### Training (to be run on target cluster)
- `cc_scripts/train_openmath.py` — CurriculumPPOTrainer + DynamicHintRLVRWorkflow (the HintRL method)
- `cc_scripts/train_openmath_opsd.py` — OPSDTrainer + OPSDWorkflow (self-distillation baseline)
- `cc_scripts/configs/openmath_hint_rl.yaml` — HintRL training config
- `cc_scripts/configs/openmath_{questa,dapo,opsd}.yaml` — 3 baseline configs
- `cc_scripts/slurm/train_*.sh` — **hardcoded to chanb/Vulcan, need Rorqual adaptation**

### Eval (working end-to-end on Rorqual)
- `cc_scripts/eval_math.py` — math eval script (entry point)
- `cc_scripts/configs/eval/eval_math.yaml` — currently configured with
  `mem_fraction_static: 0.75`, `max_running_requests: 32`
- `cc_scripts/slurm/eval_heldout_math.sh` — submit batch of eval jobs from .dat file
- `cc_scripts/slurm/resubmit_heldout_eval.sh` — (re)submit jobs from a .dat file
- `cc_scripts/slurm/salloc_eval.sh` — interactive single-dataset eval in salloc session
- `cc_scripts/slurm/eval_heldout-base_model.dat` — 2 base models × 4 datasets
- `cc_scripts/slurm/eval_heldout-hintrl.dat` — HintRL × 4 datasets
- `cc_scripts/slurm/eval_heldout-questa.dat` — QuestA × 4 datasets

### Dataset generation (NOT YET RUN for HintRL training)
- `cc_scripts/create_hint_dataset_openmath.py` → `process.py` → `convert2hf.py`
- `cc_scripts/make_hint_sweep.sh` — static 0-100% hint sweep (already generated on
  Rorqual: `/scratch/fengdic/datasets/questa/data/openr1_0..openr1_100`)

### Results & plots
- `cc_scripts/results/eval_heldout/*.jsonl` — 5-problem samples of every eval run
  (full results in scratch, which may not migrate)
- `cc_scripts/results/eval_heldout/*.yaml` — full configs for each eval run
- `cc_scripts/plots/plot_heldout_bar.py` — comparison bar chart generator
- `cc_scripts/plots/figures/heldout_eval_bar.png` — latest generated figure
- [HINTRL_EVAL_RESULTS.md](HINTRL_EVAL_RESULTS.md) — full results tables & findings

### Engine patches
- `areal/engine/sglang_remote.py` — SGLang GLIBC patch (`_patch_sglang_get_device_memory`)
- `areal/dataset/heldout_math.py` — fixed answer extraction + OlympiadBench loader
- `areal/dataset/__init__.py` — added `olympiad_bench` to VALID_DATASETS

---

## Next Steps (in priority order)

### Immediate (post-migration)
1. **Smoke test eval pipeline on new cluster**:
   ```bash
   N_SAMPLES=2 MAX_CONCURRENT=4 bash cc_scripts/slurm/salloc_eval.sh aime24 \
       <new_scratch>/models/OpenMath-Nemotron-1.5B
   ```
   Should run in < 5 min and produce 30 JSONL files with non-zero rewards.

2. **Re-port the 4 base model eval runs on new cluster** to verify pass@1 within ~2%
   of existing numbers (sanity check that eval pipeline is faithful).

3. **Finish QuestA OlympiadBench**: Either split dataset or reduce n_samples to 16.

### Short-term
4. **Generate `openr1_hint_sep` dataset** using `create_hint_dataset_openmath.py` +
   `process.py` + `convert2hf.py`. Needed for HintRL retraining.
5. **Adapt training SLURM scripts** — copy pattern from `cc_scripts/slurm/eval_heldout_math.sh`
   (already Rorqual-adapted) to train_*.sh.
6. **Run initial HintRL training** with current config on new cluster.
7. **Push new checkpoint to HF** via `push_to_hf.sh` and redo heldout eval.

### Longer-term
8. **Hyperparam sweep on Goldilocks zone** (`hyperparam_goldilock.sh`) — explore
   adaptive hint percentage curriculum.
9. **QuestA/DAPO/OPSD baseline retraining** — compare all 4 methods end-to-end.
10. **OpenCode dataset + code domain experiments**.

---

## Memory / Storage Footprint (for migration planning)

| Artifact | Location | Size |
|----------|----------|------|
| Repo | `/home/fengdic/evan_workspace/hint_rl` | ~500 MB (with .venv excluded) |
| venv | `/home/fengdic/evan_workspace/hint_rl/.venv` | ~10 GB (regenerate with `uv sync`) |
| Models (4) | `/home/fengdic/scratch/models/` | ~12 GB total |
| HintRL JSONL results | `/scratch/fengdic/hint_rl_results/.../hintrl` | ~340 MB |
| QuestA JSONL results | `/scratch/fengdic/hint_rl_results/.../questa` | ~500 MB |
| Base model JSONL results | `/scratch/fengdic/hint_rl_results/.../*_base` | ~700 MB |
| Sampled results (in repo) | `cc_scripts/results/eval_heldout/` | ~40 MB |

**Migration recommendation**: repo + sampled results fit comfortably. Full JSONL
results on scratch are likely NOT worth migrating — re-run the evals on new cluster
if you need full per-sample data, since they take only ~20 min per AIME/HMMT job
(OlympiadBench is the only one that needs special handling).

---

## Related docs

- [HINTRL_EVAL_RESULTS.md](HINTRL_EVAL_RESULTS.md) — detailed eval numbers & findings
- [CORE_KNOWLEDGE.md](CORE_KNOWLEDGE.md) — comprehensive codebase reference
- `CLAUDE.md` (repo root) — project overview + conventions for Claude Code

---

## Contacts

- Bryan Chan (`chanb@ualberta.ca`) — original codebase author, works on Vulcan cluster
- Hint RL HF repo: `FengdiFlo/HintRL`
