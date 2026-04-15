#!/bin/bash
# salloc_eval.sh - Get an interactive GPU allocation and run a single heldout eval
#
# Step 1: Get a GPU allocation
#   salloc --account=def-ashique --time=1:00:00 --mem=64G --cpus-per-task=4 --gres=gpu:h100:1
#
# Step 2: Inside the allocation, run this script
#   bash cc_scripts/slurm/salloc_eval.sh <dataset_name> [actor_path]
#
# Examples:
#   # Quick test: eval AIME24 (30 problems) with base model
#   bash cc_scripts/slurm/salloc_eval.sh aime24
#
#   # Eval AIME25 with a specific checkpoint
#   bash cc_scripts/slurm/salloc_eval.sh aime25 /home/fengdic/scratch/hint_rl_results/checkpoints/.../epoch10...
#
#   # Eval OlympiadBench with DeepScaleR
#   bash cc_scripts/slurm/salloc_eval.sh olympiad_bench agentica-org/DeepScaleR-1.5B-Preview
#
# Supported datasets: aime24, aime25, olympiad_bench, hmmt_feb_2025, brumo_2025
#
# Tip: Use fewer samples for quick testing:
#   N_SAMPLES=4 MAX_CONCURRENT=8 bash cc_scripts/slurm/salloc_eval.sh aime24

set -euo pipefail

DATASET_NAME="${1:?Usage: $0 <dataset_name> [actor_path]}"
ACTOR_PATH="${2:-/home/fengdic/scratch/models/OpenMath-Nemotron-1.5B}"

REPO_ROOT=${REPO_ROOT:-/home/fengdic/evan_workspace/hint_rl}
SCRATCH=${SCRATCH:-/home/fengdic/scratch}
N_SAMPLES=${N_SAMPLES:-32}
MAX_CONCURRENT=${MAX_CONCURRENT:-8}
TRIAL_NAME="salloc_eval_${DATASET_NAME}-$(date +%Y%m%d_%H%M%S)"

echo "========================================"
echo "  Interactive Heldout Eval"
echo "  Host:       $(hostname)"
echo "  GPU:        $(nvidia-smi -L 2>/dev/null | head -1 || echo 'no GPU detected')"
echo "  Dataset:    $DATASET_NAME"
echo "  Model:      $ACTOR_PATH"
echo "  Samples:    $N_SAMPLES"
echo "  Concurrent: $MAX_CONCURRENT"
echo "  Trial:      $TRIAL_NAME"
echo "========================================"

module load StdEnv/2023
module load cuda/12.9



source ${REPO_ROOT}/.venv/bin/activate

# Compute nodes have no internet — use HF cache only
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python ${REPO_ROOT}/cc_scripts/eval_math.py \
    --config ${REPO_ROOT}/cc_scripts/configs/eval/eval_math.yaml \
    cluster.fileroot=${SCRATCH}/hint_rl_results \
    cluster.name_resolve.nfs_record_root=${SCRATCH}/hint_rl_results/name_resolve \
    stats_logger.tensorboard.path=${SCRATCH}/hint_rl_results/logs/tensorboard \
    experiment_name=eval_${DATASET_NAME} \
    trial_name=${TRIAL_NAME} \
    actor.path=${ACTOR_PATH} \
    valid_dataset.path=${DATASET_NAME} \
    gconfig.n_samples=${N_SAMPLES} \
    gconfig.temperature=0.7 \
    gconfig.top_p=0.95 \
    rollout.max_concurrent_rollouts=${MAX_CONCURRENT}

echo ""
echo "========================================"
echo "  Done: $(date)"
echo "  Results: ${SCRATCH}/hint_rl_results/logs/fengdic/eval_${DATASET_NAME}/${TRIAL_NAME}/rollout/0/"
echo "========================================"
