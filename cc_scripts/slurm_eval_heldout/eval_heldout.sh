#!/bin/bash
# eval_heldout.sh - Submit heldout math evaluation jobs (AIME24, AIME25, OlympiadBench, HMMT, BRUMO)
# Usage: dat_file=<path_to_dat_file> bash eval_heldout.sh
#
# Each line in the dat file should export: trial_name, actor_path, dataset_name
# Example:
#   export trial_name=eval_aime24-nemotron actor_path=nvidia/OpenMath-Nemotron-1.5B dataset_name=aime24

DAT_FILE=${dat_file}
NUM_TASKS=$(wc -l < "$DAT_FILE")

# Configurable via environment variables (with defaults)
REPO_ROOT=${REPO_ROOT:-/home/fengdic/evan_workspace/hint_rl}
SCRATCH=${SCRATCH:-/home/fengdic/scratch}
ACCOUNT=${ACCOUNT:-def-ashique}
N_SAMPLES=${N_SAMPLES:-32}
MAX_CONCURRENT=${MAX_CONCURRENT:-48}
GPU_TYPE=${GPU_TYPE:-h100}
TIME_LIMIT=${TIME_LIMIT:-12:00:00}

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

  prev_job_id=$(sbatch --parsable \
    --account=${ACCOUNT} \
    --time=${TIME_LIMIT} \
    --mem=32GB \
    --cpus-per-task=2 \
    --gres=gpu:${GPU_TYPE}:1 \
    --output=${SCRATCH}/logs/hint_rl/%j.out \
    <<EOF
#!/bin/bash
module load StdEnv/2023
module load python/3.10.13
module load cuda/12.9
source ${REPO_ROOT}/.venv/bin/activate

$(sed -n "${i}p" < "$DAT_FILE")
echo "Task index: $i"
echo "Running on hostname \$(hostname)"
echo "Evaluating \${dataset_name} with model \${actor_path}"

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
EOF
)

  echo "Submitted task $i / $NUM_TASKS (job ID: $prev_job_id)"
done
