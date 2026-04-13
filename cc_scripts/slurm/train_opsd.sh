#!/bin/bash
#SBATCH --account=aip-schuurma
#SBATCH --time=06:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --array=1-1
#SBATCH --output=/home/chanb/scratch/logs/hint_rl/%j.out

module load StdEnv/2023
module load python/3.10.13
module load cuda/12.9
source /home/chanb/research/hint_rl/hint_rl/.venv/bin/activate

# python /home/chanb/research/hint_rl/hint_rl/cc_scripts/train_openmath_opsd.py \
#     --config /home/chanb/research/hint_rl/hint_rl/cc_scripts/configs/train/openmath_opsd.yaml

# python /home/chanb/research/hint_rl/hint_rl/cc_scripts/train_openmath_opsd.py \
#     --config /home/chanb/research/hint_rl/hint_rl/cc_scripts/configs/train/openmath_opsd.yaml \
#     train_dataset.path=/home/chanb/scratch/datasets/questa/data/openr1_hint_sep-small \
#     train_dataset.batch_size=8 \
#     experiment_name=debug-openmath-opsd \
#     trial_name=debug \
#     rollout.max_concurrent_rollouts=16 \
#     rollout.queue_size=16 \
#     allocation_mode=sglang:d1p1t1+d1


python /home/chanb/research/hint_rl/hint_rl/cc_scripts/train_openmath_opsd.py \
    --config /home/chanb/research/hint_rl/hint_rl/cc_scripts/configs/train/openmath_opsd.yaml \
    train_dataset.path=/home/chanb/scratch/datasets/questa/data/openr1_hint_sep-small \
    train_dataset.batch_size=8 \
    experiment_name=debug-openmath-opsd \
    trial_name=debug_with_ref_inplace \
    rollout.max_concurrent_rollouts=16 \
    rollout.queue_size=16 \
    allocation_mode=sglang:d1p1t1+d1