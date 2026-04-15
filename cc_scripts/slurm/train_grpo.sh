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

python /home/chanb/research/hint_rl/hint_rl/cc_scripts/train_openmath.py \
    --config /home/chanb/research/hint_rl/hint_rl/cc_scripts/configs/train/openmath_grpo.yaml


TRITON_PTXAS_PATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/12.9.1/bin/ptxas 


python /home/chanb/research/hint_rl/hint_rl/cc_scripts/train_openmath.py \
    --config /home/chanb/research/hint_rl/hint_rl/cc_scripts/configs/train/openmath_grpo.yaml \
    stats_logger.tensorboard.path=/home/chanb/links/scratch/hint_rl_results/logs/tensorboard \
    train_dataset.path=/home/chanb/links/scratch/datasets/questa/data/openr1_hint_sep \
    cluster.fileroot=/home/chanb/links/scratch/hint_rl_results \
    cluster.name_resolve.nfs_record_root=/home/chanb/links/scratch/hint_rl_results/name_resolve \
    actor.path=/home/chanb/links/scratch/models/OpenMath-Nemotron-1.5B \
    allocation_mode=sglang:d1+d1 \
    rollout.max_concurrent_rollouts=32 \
    train_dataset.batch_size=16