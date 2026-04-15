#!/bin/bash
# eval_heldout_math.sh - Submit heldout math evaluation jobs
# Usage: dat_file=<path_to_dat_file> bash eval_heldout_math.sh
#
# Each line in the dat file should export: trial_name, actor_path, dataset_name
# Example:
#   export trial_name=eval_aime24-nemotron_base actor_path=nvidia/OpenMath-Nemotron-1.5B dataset_name=aime24

DAT_FILE=${dat_file:?Set dat_file=<path> before running}
NUM_TASKS=$(wc -l < "$DAT_FILE")

# Configurable via environment variables (with Rorqual defaults)
REPO_ROOT=${REPO_ROOT:-/home/fengdic/evan_workspace/hint_rl}
SCRATCH=${SCRATCH:-/home/fengdic/scratch}
ACCOUNT=${ACCOUNT:-def-ashique}
N_SAMPLES=${N_SAMPLES:-32}
MAX_CONCURRENT=${MAX_CONCURRENT:-8}
GPU_TYPE=${GPU_TYPE:-h100}
TIME_LIMIT=${TIME_LIMIT:-12:00:00}

# Create log directory
LOG_DIR="${SCRATCH}/logs/hint_rl"
mkdir -p "$LOG_DIR"

echo "=== Heldout Math Evaluation ==="
echo "  dat_file:  $DAT_FILE"
echo "  tasks:     $NUM_TASKS"
echo "  account:   $ACCOUNT"
echo "  gpu:       $GPU_TYPE"
echo "  n_samples: $N_SAMPLES"
echo "  logs:      $LOG_DIR"
echo ""

wait_for_job_to_start() {
  local job_id=$1
  echo "Waiting for job $job_id to start..."
  while true; do
    local state=$(squeue -j "$job_id" -h -o "%T" 2>/dev/null)
    if [[ "$state" == "RUNNING" ]]; then
      echo "Job $job_id is running. Waiting 30 seconds before next submission..."
      sleep 30
      break
    elif [[ -z "$state" ]]; then
      echo "Job $job_id not found in queue (may have finished already). Proceeding..."
      break
    fi
    sleep 5
  done
}

prev_job_id=""

for i in $(seq 1 $NUM_TASKS); do
  if [[ -n "$prev_job_id" ]]; then
    wait_for_job_to_start "$prev_job_id"
  fi

  # Read task line to show what we're submitting
  task_line=$(sed -n "${i}p" < "$DAT_FILE")
  echo "[Task $i/$NUM_TASKS] $task_line"

  prev_job_id=$(sbatch --parsable \
    --account=${ACCOUNT} \
    --time=${TIME_LIMIT} \
    --mem=64GB \
    --cpus-per-task=4 \
    --gres=gpu:${GPU_TYPE}:1 \
    --output=${LOG_DIR}/%j-eval_heldout.out \
    --error=${LOG_DIR}/%j-eval_heldout.err \
    <<EOF
#!/bin/bash
set -euo pipefail

echo "========================================"
echo "  Heldout Eval Job"
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

$(sed -n "${i}p" < "$DAT_FILE")

echo "  trial_name:  \${trial_name}"
echo "  actor_path:  \${actor_path}"
echo "  dataset:     \${dataset_name}"
echo "  n_samples:   ${N_SAMPLES}"
echo "========================================"

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

echo "========================================"
echo "  Finished: \$(date)"
echo "========================================"
EOF
)

  echo "  Submitted job $prev_job_id → log: ${LOG_DIR}/${prev_job_id}-eval_heldout.out"
  echo ""
done

echo "=== All $NUM_TASKS jobs submitted ==="
echo "Monitor: squeue -u \$(whoami)"
echo "Logs:    tail -f ${LOG_DIR}/*-eval_heldout.out"
