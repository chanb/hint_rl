To build apptainer for CC:
```
module load StdEnv/2023
module load python/3.10.13
module load cuda/12.9
module load apptainer/1.4.5
export APPTAINER_CACHEDIR=~/scratch
apptainer build ~/scratch/questa.sif docker://ghcr.io/inclusionai/areal-runtime:v1.0.1
```

Run the apptainer interactively:
```
module load StdEnv/2023
module load python/3.10.13
module load cuda/12.9
module load apptainer/1.4.5

apptainer run --nv -C -W $SLURM_TMPDIR -B ~/research/hint_rl:/workspace -B ~/scratch/datasets:/datasets -B ~/scratch/hint_rl_results:/hint_rl_results --writable-tmpfs ~/scratch/questa.sif

cd /workspace/hint_rl
uv pip install --no-deps -e .
uv pip install ipdb
```

Run code:
```
# Local scheduler
cd /workspace/hint_rl
python3 cc_scripts/openmath_rl.py --config cc_scripts/openmath_questa_50_grpo.yaml scheduler.type=local

# Ray scheduler
ray start --head --disable-usage-stats
python3 cc_scripts/openmath_rl.py --config cc_scripts/openmath_questa_50_grpo.yaml scheduler.type=ray
```

Run with `uv`:
```
module load StdEnv/2023
module load python/3.12.4
module load cuda/12.9

uv sync --extra cuda # One-time only
source /home/chanb/research/hint_rl/hint_rl/.venv/bin/activate

# Test
python /home/chanb/research/hint_rl/hint_rl/cc_scripts/openmath_rl.py --config /home/chanb/research/hint_rl/hint_rl/cc_scripts/openmath_questa_50_grpo.yaml

# Actual run with similar hyperparameters as QuestA
python /home/chanb/research/hint_rl/hint_rl/cc_scripts/openmath_rl.py --config /home/chanb/research/hint_rl/hint_rl/cc_scripts/openmath_questa_50_grpo_slurm.yaml
```

Tensorboard:
```
Compute node $: tensorboard --logdir=. --host 0.0.0.0 --load_fast false

Local $: ssh -N -f -L localhost:6007:<node_name>:6006 <username>@vulcan.alliancecan.ca
```

Process data (within apptainer) as mentioned in paper:
```
mkdir -p /datasets/questa/data/

cd /workspace/hint_rl/cc_scripts/datasets

# Make training set
python add_prefix.py --data_path=/datasets/questa/OpenR1-50-0-4.jsonl --out_path=/datasets/questa/OpenR1-50-0-4-prefix.jsonl --ratio=50
python process.py --input=/datasets/questa/OpenR1-50-0-4-prefix.jsonl --output=/datasets/questa/data/train.jsonl

# Make test set
python add_prefix.py --data_path=/datasets/questa/OpenR1-50-0-4.jsonl --out_path=/datasets/questa/OpenR1-0-0-4-prefix.jsonl --ratio=0
python process.py --input=/datasets/questa/OpenR1-0-0-4-prefix.jsonl --output=/datasets/questa/data/test.jsonl

# Make HF dataset
python convert2hf.py
```

Pretrained models:
Qwen 1.5B: `deepseek-ai/DeepSeek-R1-Distill-1.5B`
Default Nemotron: `nvidia/OpenMath-Nemotron-1.5B`
From paper: `QuestA/QuestA-Nemotron-1.5B`
