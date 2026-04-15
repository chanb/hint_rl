#!/bin/bash
#SBATCH --account=def-schuurma
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=h100:2
#SBATCH --array=1-1
#SBATCH --output=/home/chanb/scratch/logs/hint_rl/%j.out

module load StdEnv/2023
module load python/3.10.13
module load cuda/12.9
source /home/chanb/research/hint_rl/hint_rl/.venv/bin/activate

python /home/chanb/research/hint_rl/hint_rl/cc_scripts/train_openmath.py \
    --config /home/chanb/research/hint_rl/hint_rl/cc_scripts/configs/train/openmath_grpo.yaml
