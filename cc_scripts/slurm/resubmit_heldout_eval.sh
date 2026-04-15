#!/bin/bash
# resubmit_heldout_eval.sh - Cancel existing heldout eval jobs and resubmit all
#
# Usage:
#   bash cc_scripts/slurm/resubmit_heldout_eval.sh
#   bash cc_scripts/slurm/resubmit_heldout_eval.sh --no-cancel   # submit without cancelling
#   bash cc_scripts/slurm/resubmit_heldout_eval.sh --dry-run     # show what would be submitted

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/fengdic/evan_workspace/hint_rl}
SCRATCH=${SCRATCH:-/home/fengdic/scratch}
ACCOUNT=${ACCOUNT:-def-ashique}
N_SAMPLES=${N_SAMPLES:-32}
MAX_CONCURRENT=${MAX_CONCURRENT:-8}
GPU_TYPE=${GPU_TYPE:-h100}
TIME_LIMIT=${TIME_LIMIT:-12:00:00}
MEM=${MEM:-64G}
DAT_FILE=${DAT_FILE:-${REPO_ROOT}/cc_scripts/slurm/eval_heldout-base_model.dat}
LOG_DIR="${SCRATCH}/logs/hint_rl"

DRY_RUN=false
NO_CANCEL=false
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true ;;
        --no-cancel) NO_CANCEL=true ;;
    esac
done

mkdir -p "$LOG_DIR"

echo "=== Heldout Eval (Re)submit ==="
echo "  dat_file:     $DAT_FILE"
echo "  account:      $ACCOUNT"
echo "  gpu:          $GPU_TYPE"
echo "  n_samples:    $N_SAMPLES"
echo "  max_concurrent: $MAX_CONCURRENT"
echo "  time_limit:   $TIME_LIMIT"
echo ""

# Cancel existing heldout eval jobs
if [[ "$NO_CANCEL" == false ]]; then
    existing=$(squeue -u $(whoami) -o "%.10i %.50j" -h 2>/dev/null | grep "eval_heldout" | awk '{print $1}' || true)
    if [[ -n "$existing" ]]; then
        echo "Cancelling existing eval_heldout jobs: $existing"
        if [[ "$DRY_RUN" == false ]]; then
            echo "$existing" | xargs scancel
            sleep 2
        fi
    else
        echo "No existing eval_heldout jobs found."
    fi
fi

# Submit all jobs from dat file
NUM_TASKS=$(wc -l < "$DAT_FILE")
echo ""
echo "Submitting $NUM_TASKS jobs..."

for i in $(seq 1 $NUM_TASKS); do
    task_line=$(sed -n "${i}p" < "$DAT_FILE")
    echo "  [$i/$NUM_TASKS] $task_line"

    if [[ "$DRY_RUN" == true ]]; then
        continue
    fi

    job_id=$(sbatch --parsable \
        --job-name="eval_heldout_${i}" \
        --account=${ACCOUNT} \
        --time=${TIME_LIMIT} \
        --mem=${MEM} \
        --cpus-per-task=4 \
        --gres=gpu:${GPU_TYPE}:1 \
        --output=${LOG_DIR}/%j-eval_heldout.out \
        --error=${LOG_DIR}/%j-eval_heldout.err \
        <<EOF
#!/bin/bash
set -euo pipefail
echo "========================================"
echo "  Host:    \$(hostname)"
echo "  GPU:     \$(nvidia-smi -L 2>/dev/null | head -1)"
echo "  Started: \$(date)"
echo "========================================"
module load StdEnv/2023
module load cuda/12.9

source ${REPO_ROOT}/.venv/bin/activate

# Compute nodes have no internet — use HF cache only
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

${task_line}
echo "  trial:   \${trial_name}"
echo "  actor:   \${actor_path}"
echo "  dataset: \${dataset_name}"

python ${REPO_ROOT}/cc_scripts/eval_math.py \
  --config ${REPO_ROOT}/cc_scripts/configs/eval/eval_math.yaml \
  cluster.fileroot=${SCRATCH}/hint_rl_results \
  cluster.name_resolve.nfs_record_root=${SCRATCH}/hint_rl_results/name_resolve \
  stats_logger.tensorboard.path=${SCRATCH}/hint_rl_results/logs/tensorboard \
  experiment_name=eval_\${dataset_name} \
  trial_name=\${trial_name} \
  actor.path=\${actor_path} \
  valid_dataset.path=\${dataset_name} \
  gconfig.n_samples=${N_SAMPLES} \
  gconfig.temperature=0.7 \
  gconfig.top_p=0.95 \
  rollout.max_concurrent_rollouts=${MAX_CONCURRENT}

echo "Finished: \$(date)"
EOF
    )
    echo "           → job $job_id"
done

echo ""
if [[ "$DRY_RUN" == true ]]; then
    echo "=== DRY RUN — no jobs submitted ==="
else
    echo "=== All $NUM_TASKS jobs submitted ==="
    echo "Monitor:  squeue -u $(whoami)"
    echo "Logs:     tail -f ${LOG_DIR}/*-eval_heldout.out"
fi
