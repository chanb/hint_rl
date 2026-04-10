#!/bin/bash
#SBATCH --account=aip-schuurma
#SBATCH --time=72:00:00
#SBATCH --mem=400GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --array=1-1
#SBATCH --output=/home/chanb/scratch/logs/hint_rl/%j.out

module load StdEnv/2023
module load python/3.10.13
module load cuda/12.9
source /home/chanb/research/hint_rl/hint_rl/.venv/bin/activate

python /home/chanb/research/hint_rl/hint_rl/cc_scripts/train_openmath_opsd.py --config /home/chanb/research/hint_rl/hint_rl/cc_scripts/configs/train/openmath_opsd.yaml