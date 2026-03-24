#!/bin/bash
# submit_jobs.sh - submits one job per line in the dat file, each on a unique node

DAT_FILE=${dat_file}
NUM_TASKS=$(wc -l < "$DAT_FILE")

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
  # Wait for previous job to start before submitting the next
  if [[ -n "$prev_job_id" ]]; then
    wait_for_job_to_start "$prev_job_id"
  fi

  prev_job_id=$(sbatch --parsable \
    --account=aip-schuurma \
    --time=12:00:00 \
    --mem=32GB \
    --cpus-per-task=2 \
    --gres=gpu:l40s:1 \
    --output=/home/chanb/scratch/logs/hint_rl/%j.out \
    <<EOF
#!/bin/bash
module load StdEnv/2023
module load python/3.10.13
module load cuda/12.9
source /home/chanb/research/hint_rl/hint_rl/.venv/bin/activate

$(sed -n "${i}p" < "$DAT_FILE")
echo "Task index: $i"
echo "Running on hostname \$(hostname)"

python /home/chanb/research/hint_rl/hint_rl/cc_scripts/olympiad_bench_eval.py \
  --config /home/chanb/research/hint_rl/hint_rl/cc_scripts/eval-questa_50.yaml \
  trial_name=\${trial_name} \
  actor.path=\${actor_path} \
  valid_dataset.path=\${dataset_path}
EOF
)

  echo "Submitted task $i / $NUM_TASKS (job ID: $prev_job_id)"
done