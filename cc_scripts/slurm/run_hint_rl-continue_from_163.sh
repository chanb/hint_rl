#!/bin/bash
#SBATCH --account=aip-schuurma
#SBATCH --time=48:00:00
#SBATCH --mem=200GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:4
#SBATCH --array=1-1
#SBATCH --output=/home/chanb/scratch/logs/hint_rl/%j.out

module load StdEnv/2023
module load python/3.10.13
module load cuda/12.9
source /home/chanb/research/hint_rl/hint_rl/.venv/bin/activate

python /home/chanb/research/hint_rl/hint_rl/cc_scripts/openmath_rl.py --config /home/chanb/research/hint_rl/hint_rl/cc_scripts/openmath_hint_rl_grpo.yaml actor.path=/home/chanb/scratch/hint_rl_results/checkpoints/chanb/openmath-dapo/local_train-hint_rl/default/epoch11epochstep9globalstep163 trial_name=local_train-hint_rl-continue_from_163