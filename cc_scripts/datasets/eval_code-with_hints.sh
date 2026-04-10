#!/bin/bash
#SBATCH --account=aip-schuurma
#SBATCH --time=12:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=/home/chanb/scratch/logs/hint_rl/%j.out

module load StdEnv/2023
module load python/3.10.13
module load cuda/12.9
source /home/chanb/research/hint_rl/hint_rl/.venv/bin/activate

python /home/chanb/research/hint_rl/hint_rl/cc_scripts/eval_code-with_hints.py \
    --config /home/chanb/research/hint_rl/hint_rl/cc_scripts/configs/eval/eval_code.yaml \
    trial_name=local_eval-evaluate_code-50_hints-openreasoning-nemotron \
    actor.path=nvidia/OpenReasoning-Nemotron-1.5B \
    valid_dataset.path=/home/chanb/scratch/datasets/opencode/data/opencode_hint_sep \
    rollout.max_concurrent_rollouts=20
