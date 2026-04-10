#!/bin/bash
#SBATCH --account=aip-schuurma
#SBATCH --time=00:48:00
#SBATCH --mem=200GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:4
#SBATCH --array=1-4
#SBATCH --output=/home/chanb/scratch/logs/hint_rl/%j.out

module load StdEnv/2023
module load python/3.10.13
module load cuda/12.9
source /home/chanb/research/hint_rl/hint_rl/.venv/bin/activate

`sed -n "${SLURM_ARRAY_TASK_ID}p" < /home/chanb/research/hint_rl/hint_rl/cc_scripts/slurm/hyperparam_goldilock.dat`
echo ${SLURM_ARRAY_TASK_ID}

python /home/chanb/research/hint_rl/hint_rl/cc_scripts/train_openmath.py --config /home/chanb/research/hint_rl/hint_rl/cc_scripts/configs/train/openmath_hint_rl.yaml \
    total_train_epochs=25 \
    experiment_name=hyperparam_sweep-goldilock \
    trial_name=${trial_name} \
    dynamic_hint.initial_hint=${initial_hint} \
    dynamic_hint.hint_delta=${hint_delta} \
    dynamic_hint.goldilock_zone=${goldilock_zone}
