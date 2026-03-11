#!/bin/bash
#SBATCH --account=aip-schuurma
#SBATCH --time=48:00:00
#SBATCH --mem=100GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:4
#SBATCH --array=1-1
#SBATCH --output=/home/chanb/scratch/logs/hint_rl/%j.out

module load StdEnv/2023
module load python/3.10.13
module load cuda/12.9
module load apptainer/1.4.5

apptainer run --nv -C -W $SLURM_TMPDIR -B ~/research/hint_rl:/workspace -B ~/scratch/datasets:/datasets -B ~/scratch/hint_rl_results:/hint_rl_results --writable-tmpfs ~/scratch/questa.sif bash /workspace/hint_rl/cc_scripts/slurm/run_questa_50.sh