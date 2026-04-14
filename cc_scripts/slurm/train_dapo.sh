#!/bin/bash
#SBATCH --account=aip-schuurma
#SBATCH --time=24:00:00
#SBATCH --mem=100GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --array=1-1
#SBATCH --output=/home/chanb/scratch/logs/hint_rl/%j.out

module load StdEnv/2023
module load python/3.10.13
module load cuda/12.9
source /home/chanb/research/hint_rl/hint_rl/.venv/bin/activate

# First run---stopped at 128 steps and took 2 days and 3 hours (rounded 1 hour up)---last checkpoint is Epoch 9/50 Step 14/14 Train step 126/700
# python /home/chanb/research/hint_rl/hint_rl/cc_scripts/train_openmath.py --config /home/chanb/research/hint_rl/hint_rl/cc_scripts/configs/train/openmath_dapo.yaml

# Second run (before recover handler is specified...)---this run has recover handler so can rerun command
python /home/chanb/research/hint_rl/hint_rl/cc_scripts/train_openmath.py \
    --config /home/chanb/research/hint_rl/hint_rl/cc_scripts/configs/train/openmath_dapo.yaml \
    actor.path=/home/chanb/scratch/hint_rl_results/checkpoints/chanb/openmath-dapo/local_train/default/epoch8epochstep13globalstep125 \
    trial_name=load_from_globalstep_126 \
    total_train_epochs=41 \
    seed=2
