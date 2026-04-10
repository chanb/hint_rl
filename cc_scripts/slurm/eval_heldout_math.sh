#!/bin/bash
#SBATCH --account=aip-schuurma
#SBATCH --time=01:00:00
#SBATCH --mem=20GB
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --array=1-1
#SBATCH --output=/home/chanb/scratch/logs/hint_rl/%j.out

module load StdEnv/2023
module load python/3.10.13
module load cuda/12.9
source /home/chanb/research/hint_rl/hint_rl/.venv/bin/activate

export actor_path="TODO"
export n_samples=32
export max_concurrent_rollouts=4

dataset_names=(
    aime24
    aime25
    brumo_2025
    hmmt_feb_2025
)

for dataset_name in "${dataset_names[@]}"; do
    echo "$dataset_name"
    python /home/bryanpu1/projects/neurips_2026/hint_rl/cc_scripts/eval_math.py \
        --config /home/bryanpu1/projects/neurips_2026/hint_rl/cc_scripts/configs/eval/eval_math.yaml \
        cluster.fileroot=/home/bryanpu1/projects/neurips_2026/scratch \
        cluster.name_resolve.nfs_record_root=/home/bryanpu1/projects/neurips_2026/scratch/name_resolve \
        stats_logger.tensorboard.path=/home/bryanpu1/projects/neurips_2026/scratch/tensorboard \
        allocation_mode=sglang:d8p1t1 \
        gconfig.top_p=0.95 \
        gconfig.temperature=0.7 \
        gconfig.n_samples=${n_samples} \
        rollout.max_concurrent_rollouts=${max_concurrent_rollouts} \
        valid_dataset.path=${dataset_name} \
        experiment_name=eval_${dataset_name} \
        actor.path=${actor_path}
done