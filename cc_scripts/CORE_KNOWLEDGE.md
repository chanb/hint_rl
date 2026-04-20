# Hint RL — Core Knowledge Document

## 1. Project Overview

This project implements **Hint-based Reinforcement Learning** for training LLMs on math
and code reasoning tasks. Built on top of the **AReaL** distributed RL framework.

**Core idea**: Instead of raw RL (DAPO) or fixed hint schedules (QuestA), dynamically
adjust the percentage of solution hints given to the model during training based on a
"Goldilocks zone" — a target success rate range where learning is most effective.

**Base model**: `nvidia/OpenMath-Nemotron-1.5B`

---

## 2. Architecture

### Training Pipeline
```
Dataset → Workflow (hint injection) → Engine (SGLang rollout + FSDP training) → Trainer (PPO + curriculum)
```

### Key Components

| Component | Class | File |
|-----------|-------|------|
| Main trainer | `CurriculumPPOTrainer` | `areal/trainer/rl_trainer.py` |
| OPSD trainer | `OPSDTrainer` | `areal/trainer/rl_trainer.py` |
| Hint workflow | `DynamicHintRLVRWorkflow` | `areal/workflow/dynamic_hint_rlvr.py` |
| OPSD workflow | `OPSDWorkflow` | `areal/workflow/opsd.py` |
| Math reward | `gsm8k_reward_fn` → `MathVerifyWorker` | `areal/reward/gsm8k.py`, `areal/reward/__init__.py` |
| Code reward | `opencode_reward_fn` → `CodeVerifyWorker` | `areal/reward/opencode.py`, `areal/reward/__init__.py` |
| Dataset registry | `get_custom_dataset()` | `areal/dataset/__init__.py` |
| Heldout math | AIME24/25, OlympiadBench, HMMT, BRUMO | `areal/dataset/heldout_math.py` |

### Hint Curriculum System (CurriculumPPOTrainer)

Two modes:
1. **Adaptive (Goldilocks)**: Per-sample hint percentage adjusted based on success rate
   - If success_rate > upper_bound → reduce hint by `hint_delta`
   - If success_rate < lower_bound → increase hint by `hint_delta`
   - Stored as per-sample dict in `workflow_kwargs["hint_percentage"]`

2. **Fixed schedule**: Pre-defined hint percentage changes at specific steps
   - e.g., QuestA: 50% for 100 steps → 25% for remaining

### OPSD (On-Policy Self-Distillation)
- Generates two outputs per sample: with and without hints
- Uses hint logprobs for distillation loss (no explicit reward signal)
- `OPSDTrainer` calls `compute_opsd_advantages()` instead of standard advantage

---

## 3. Experiments (4 conditions)

| Condition | Config | Trainer | Initial Hint | Schedule |
|-----------|--------|---------|--------------|----------|
| **Hint RL (ours)** | `openmath_hint_rl.yaml` | CurriculumPPOTrainer | 100% | Adaptive [0.5, 0.75] zone, delta=10 |
| **QuestA** | `openmath_questa.yaml` | CurriculumPPOTrainer | 50% | Fixed: 25% at step 100 |
| **DAPO** | `openmath_dapo.yaml` | CurriculumPPOTrainer | 0% | None (no hints) |
| **OPSD** | `openmath_opsd.yaml` | OPSDTrainer | 100% | Static |

All use:
- 50 epochs, batch size 128, lr=2e-5, Adam
- Generation: n_samples=8, max_new_tokens=12000, temperature=1.0
- Allocation: `sglang:d2p1t1+fsdp:d2` (2 GPUs rollout, 2 GPUs training)
- Dataset: `openr1_hint_sep` (OpenR1 with separated hint column)

### Hyperparameter Sweep (Goldilocks Zone)
- 4 configurations tested via `hyperparam_goldilock.sh` array job
- Sweeps: goldilock_zone bounds, initial_hint, hint_delta
- Short runs: 25 epochs each
- See `hyperparam_goldilock.dat` for parameter combos

---

## 4. Dataset Pipeline

### Math (OpenR1 → Hint-separated)
```
OpenR1-50-0-4.jsonl
  → create_hint_dataset_openmath.py  (split on </think>, extract prefix as hint)
  → process.py                       (normalize fields: prompt→question, solutions→answer)
  → convert2hf.py                    (convert to HuggingFace Dataset format)
  → openr1_hint_sep/                 (final training dataset)
```

### Code (OpenThoughts → Hint-separated)
```
open-r1/OpenThoughts-114k-Code_decontaminated (from HuggingFace)
  → create_hint_dataset_opencode.py  (split on "### Solution Code", validate tests)
  → process.py → convert2hf.py → opencode_hint_sep/
```

### QuestA Static Sweep (for ablations)
```
add_prefix.py --ratio=N            (create N% hint prefix)
  → process.py → convert2hf.py
  → openr1_0, openr1_10, ..., openr1_100
```
Generated via `make_hint_sweep.sh`

### Dataset Format
Training data (`openr1_hint_sep`):
```json
{"question": "...", "answer": "\\boxed{...}", "id": "N", "hint": "<partial solution>"}
```
The workflow (`DynamicHintRLVRWorkflow`) dynamically injects hints by prepending
`## Hint.\n{hint_prefix}` to the user message based on the current hint_percentage.

---

## 5. Evaluation

### Eval Script: `eval_math.py`
- Uses `RLVRWorkflow` + `gsm8k_reward_fn` for math evaluation
- SGLang engine for inference (single GPU: `sglang:d1p1t1`)
- temperature=0.7, top_p=0.95, n_samples=8
- Skips already-computed results (checks for existing `.jsonl` files)
- Results: `{fileroot}/logs/{user}/{experiment_name}/{trial_name}/rollout/0/{task_id}.jsonl`

### Eval Config: `eval_math.yaml`
- Key params to override per-eval: `trial_name`, `actor.path`, `valid_dataset.path`
- `max_concurrent_rollouts=24` for eval (vs 48 for training)

### Heldout Datasets (in `areal/dataset/heldout_math.py`)
| Dataset | HF Path | Split | Has Answer | Size |
|---------|---------|-------|------------|------|
| AIME24 | `math-ai/aime24` | test | Yes (`solution`) | 30 |
| AIME25 | `math-ai/aime25` | test | Yes (`answer`) | 30 |
| OlympiadBench | `lmms-lab/OlympiadBench` | test_en | Yes (`final_answer[0]`, 1517 after null filter) | 1517 |
| HMMT Feb 2025 | `MathArena/hmmt_feb_2025` | train | Yes (`answer`) | 30 |
| BRUMO 2025 | `MathArena/brumo_2025` | train | Yes (`answer`) | 30 |

All use system prompt: "Please reason step by step, and put your final answer within \boxed{}."

**Reward function for heldout eval**: The existing `gsm8k_reward_fn` (backed by
`MathVerifyWorker` / `math_verify`) works for all math datasets since it handles
`\boxed{}` extraction and numeric/LaTeX comparison. No new reward function needed.

### Eval SLURM Scripts

**Batch submission** — `slurm/resubmit_heldout_eval.sh`:
```bash
# Submit all (cancels existing first)
bash cc_scripts/slurm/resubmit_heldout_eval.sh

# Preview without submitting
bash cc_scripts/slurm/resubmit_heldout_eval.sh --dry-run

# Submit without cancelling existing
bash cc_scripts/slurm/resubmit_heldout_eval.sh --no-cancel

# Override settings via env vars
N_SAMPLES=8 MAX_CONCURRENT=16 bash cc_scripts/slurm/resubmit_heldout_eval.sh
```

**Interactive testing** — `slurm/salloc_eval.sh`:
```bash
# Step 1: Get a GPU
salloc --account=def-ashique --time=1:00:00 --mem=64G --cpus-per-task=4 --gres=gpu:h100:1

# Step 2: Run eval interactively (see output in real-time)
bash cc_scripts/slurm/salloc_eval.sh aime24
bash cc_scripts/slurm/salloc_eval.sh aime25 agentica-org/DeepScaleR-1.5B-Preview

# Quick smoke test (4 samples instead of 32)
N_SAMPLES=4 MAX_CONCURRENT=8 bash cc_scripts/slurm/salloc_eval.sh aime24
```

**Data file format** (`slurm/eval_heldout-base_model.dat`):
```
export trial_name=<name> actor_path=<model_or_checkpoint> dataset_name=<dataset>
```

### Heldout Eval (on Salient cluster)
Example from setup.md:
```bash
python eval_math.py --config configs/eval/eval_math.yaml \
  cluster.fileroot=<scratch> \
  allocation_mode=sglang:d8p1t1 \
  valid_dataset.path=aime24 \
  experiment_name=eval_aime24 \
  gconfig.top_p=0.95 gconfig.temperature=0.7 gconfig.n_samples=32 \
  rollout.max_concurrent_rollouts=64
```

---

## 6. Checkpointing & HuggingFace Integration

### Checkpoint Saving (during training)
- **HF format** (for inference/publishing): `{fileroot}/checkpoints/{user}/{experiment}/{trial}/default/epoch{E}epochstep{S}globalstep{G}/`
  - Saves `config.json`, `model.safetensors`, tokenizer files
  - Handled by `areal/utils/saver.py` (sync or async modes)
- **DCP format** (for training resumption): `recover_checkpoint/` directory
  - Includes model weights, optimizer state, RNG, dataloader position

### HuggingFace Push/Pull
- **Pull**: Already implemented via `areal/utils/hf_utils.py`
  - `load_hf_tokenizer()`, `download_from_huggingface()`, `load_hf_or_local_file()`
- **Push**: NOT implemented in codebase. Checkpoints are saved in HF format on disk
  but not uploaded. To push:
  ```python
  from huggingface_hub import HfApi
  api = HfApi()
  api.upload_folder(
      folder_path="<checkpoint_dir>",
      repo_id="<org>/<model_name>",
      repo_type="model",
  )
  ```
  Or use `huggingface-cli upload <org>/<model_name> <checkpoint_dir>`.

---

## 7. Cluster Configuration

### Rorqual (Alliance Canada) — Current cluster for fengdic
- **Hostname**: `rorqual3` (login node)
- **OS**: AlmaLinux 9.7
- **GPU**: NVIDIA H100 80GB HBM3, 4 per node
  - MIG slices available: 3g.40gb, 2g.20gb, 1g.10gb
- **CPU**: 64 cores per GPU node, 192 cores on login
- **RAM**: 512GB per GPU node
- **SLURM accounts**: `def-ashique`, `rrg-ashique`
- **GPU partitions**:
  | Partition | Max Time | Notes |
  |-----------|----------|-------|
  | `gpubase_interac` | 8h | Interactive, max 2 nodes |
  | `gpubase_bynode_b1` | 3h | Full node |
  | `gpubase_bynode_b2` | 12h | Full node |
  | `gpubase_bynode_b3` | 1d | Full node |
  | `gpubase_bynode_b4` | 3d | Full node |
  | `gpubase_bynode_b5` | 7d | Full node |
  | `gpubase_bygpu_b1-b5` | 3h-7d | Per-GPU allocation |
  | `gpubackfill` | 1d | Preemptible |
- **Modules**: `StdEnv/2023`, `python/3.10.13`, `python/3.12.4`, `cuda/12.2|12.6|12.9|13.2`
- **Quotas (fengdic)**:
  - Home: 50GB / 500K files
  - Scratch: 20TB / 1M files
  - Project (def-ashique): 1TB / 500K files
- **GRES syntax**: `--gres=gpu:h100:N` (NOT `gpu:l40s:N`)
- **H100 vs L40S**: H100 has 80GB VRAM (vs 48GB L40S), so `max_concurrent_rollouts`
  can be higher. Estimate ~40-50 per H100 GPU for 1.5B model.

### Vulcan (Alliance Canada) — chanb's original cluster
- SLURM account: `aip-schuurma`
- GPU type: L40S (48GB VRAM)
- Tuning: ~28 `max_concurrent_rollouts` per GPU on L40S

### Salient (alternate cluster)
- Used for heldout eval with more GPUs (`sglang:d8p1t1`, 64 concurrent rollouts)

### Key GPU Utilization Params (safe to change)
- `rollout.max_concurrent_rollouts` — parallelism for inference
- `rollout.queue_size` — rollout queue depth
- `allocation_mode` — GPU distribution between rollout and training

---

## 8. Hardcoded Paths

### Original (chanb on Vulcan) — in configs/slurm scripts
- Scratch: `/home/chanb/scratch/hint_rl_results`
- Datasets: `/home/chanb/scratch/datasets/questa/data/openr1_hint_sep`
- Venv: `/home/chanb/research/hint_rl/hint_rl/.venv/bin/activate`
- Scripts: `/home/chanb/research/hint_rl/hint_rl/cc_scripts/`
- Logs: `/home/chanb/scratch/logs/hint_rl/%j.out`
- Tensorboard: `/home/chanb/scratch/hint_rl_results/logs/tensorboard`
- SLURM: `--account=aip-schuurma`, `--gres=gpu:l40s:N`

### fengdic on Rorqual — correct mappings
- Repo: `/home/fengdic/evan_workspace/hint_rl`
- Scratch: `/home/fengdic/scratch`
- Results: `/home/fengdic/scratch/hint_rl_results`
- Datasets: `/home/fengdic/scratch/datasets/questa/data/`
- Venv: `/home/fengdic/evan_workspace/hint_rl/.venv/bin/activate`
- SLURM: `--account=def-ashique`, `--gres=gpu:h100:N`

### Scratch Contents (as of 2026-04-10)
- `datasets/questa/OpenR1-50-0-4.jsonl` — source data ✓
- `datasets/questa/data/openr1_0..openr1_100` — QuestA static sweep (HF format) ✓
- `datasets/questa/data/openr1_hint_sep` — **MISSING** (needs generation per setup.md)
- `datasets/opencode/` — **MISSING** (needs generation)
- `hint_rl_results/` — some test tensorboard logs, empty checkpoint dir from test runs

---

## 9. Visualization / Analysis

| Notebook | Purpose |
|----------|---------|
| `check_hint_change.ipynb` | Plot hint % progression over training epochs |
| `check_hint_usefulness-trained.ipynb` | Learning curves (Pass@K vs training steps) |
| `check_hint_usefulness-per_hint_percentage.ipynb` | Goldilocks zone analysis per static hint % |
| `check_hint_usefulness-per_model.ipynb` | Hint impact across models (DeepScaleR, Nemotron, QuestA, Qwen) |

---

## 10. Open Items

1. **Code domain training**: TBD in setup.md
2. **OpenCode extra filtering**: Evaluating `nvidia/OpenReasoning-Nemotron-1.5B` on code dataset (TODO in setup.md)
3. **HuggingFace push**: No automated upload — need utility script
4. **Heldout eval slurm scripts**: Need generation for AIME24/25, OlympiadBench, HMMT, BRUMO
5. **AIME25, HMMT, BRUMO**: No ground truth answers — evaluation generates responses but can't auto-score
