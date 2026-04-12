#!/bin/bash
#SBATCH --account=aip-schuurma
#SBATCH --time=24:00:00
#SBATCH --mem=50GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --array=1-1
#SBATCH --output=/home/chanb/scratch/logs/hint_rl/%j.out

module load StdEnv/2023
module load python/3.10.13
module load cuda/12.9
source /home/chanb/research/hint_rl/hint_rl/.venv/bin/activate

# First run---stopped at 103/700 steps and took 1 day and 17 hours (rounded 1 hour up)---last checkpoint is Epoch 7/50 Step 14/14 Train step 98/700
# python /home/chanb/research/hint_rl/hint_rl/cc_scripts/train_openmath.py --config /home/chanb/research/hint_rl/hint_rl/cc_scripts/configs/train/openmath_questa.yaml

# Second run (before recover handler is specified...)---this run has recover handler so can rerun command
python /home/chanb/research/hint_rl/hint_rl/cc_scripts/train_openmath.py \
    --config /home/chanb/research/hint_rl/hint_rl/cc_scripts/configs/train/openmath_questa.yaml \
    actor.path=/home/chanb/scratch/hint_rl_results/checkpoints/chanb/openmath-questa/local_train/default/epoch6epochstep13globalstep97 \
    trial_name=load_from_globalstep_98 \
    total_train_epochs=43 \
    dynamic_hint.dynamic_hint_schedule.change_steps=[2] \
    seed=2