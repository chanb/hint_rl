#!/bin/bash
#SBATCH --account=aip-schuurma
#SBATCH --time=06:00:00
#SBATCH --mem=100GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:4
#SBATCH --array=1-8
#SBATCH --output=/home/chanb/scratch/logs/hint_rl/%j.out

module load StdEnv/2023
module load python/3.10.13
module load cuda/12.9
source /home/chanb/research/hint_rl/hint_rl/.venv/bin/activate

`sed -n "${SLURM_ARRAY_TASK_ID}p" < /home/chanb/research/hint_rl/hint_rl/cc_scripts/slurm/eval_configs.dat`
echo ${SLURM_ARRAY_TASK_ID}

python /home/chanb/research/hint_rl/hint_rl/cc_scripts/olympiad_bench_eval.py --config /home/chanb/research/hint_rl/hint_rl/cc_scripts/eval-questa_50.yaml trial_name=${trial_name} actor.path=${actor_path} valid_dataset.path=${dataset_path}