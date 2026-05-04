#!/bin/bash
#SBATCH --account=def-jacobsen
#SBATCH --time=72:00:00
#SBATCH --mem=400GB
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1
#SBATCH --output=/scratch/tianyifa/logs/hint_rl/%j.out

REPO_ROOT=/home/tianyifa/hint_rl

module load StdEnv/2023
module load cuda/12.9
unset LD_PRELOAD
source ${REPO_ROOT}/.venv/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python ${REPO_ROOT}/cc_scripts/train_openmath.py \
    --config ${REPO_ROOT}/cc_scripts/configs/train/openmath_hint_rl.yaml
