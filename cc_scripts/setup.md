### First time setup
Run with `uv`:
```
# If uv isn't installed already
pipx install uv

# If pipx install uv doesn't work
curl -LsSf https://astral.sh/uv/install.sh | sh

# Load modules, do this every time before running things
module load StdEnv/2023
module load python/3.12.4
module load cuda/12.9

# One-time setup to create the virtual environment using uv
cd <PATH_TO>/hint_rl
uv sync --extra cuda

# Activate virtualenv, do this every time before running things
source <PATH_TO>/hint_rl/.venv/bin/activate
```

### Dataset
The `jsonl` files are stored in [here](https://huggingface.co/datasets/foreverlasting1202/QuestA/tree/main).
The `make_hint_sweep.sh` file will create datasets with hints from 0% to 100%, with 10% increments, stored in `<PATH_TO>/datasets/questa/data/`:
```
mkdir -p <PATH_TO>/datasets/questa/data/
cd <PATH_TO>/datasets/questa
wget https://huggingface.co/datasets/foreverlasting1202/QuestA/resolve/main/OpenR1-50-0-4.jsonl

cd <PATH_TO>/hint_rl/cc_scripts/datasets
code_path=<PATH_TO>/hint_rl dataset_path=<PATH_TO>/datasets/questa ./make_hint_sweep.sh
```

#### Hint RL dataset
```
export dataset_path=<PATH_TO>/datasets/questa

cd <PATH_TO>/hint_rl/cc_scripts/datasets

python create_hint_dataset.py --data_path=${dataset_path}/OpenR1-50-0-4.jsonl --out_path=${dataset_path}/data/OpenR1-hint_sep.jsonl
python process.py --input=${dataset_path}/data/OpenR1-hint_sep.jsonl --output=${dataset_path}/data/train-hint_sep.jsonl
python convert2hf.py --train_input=${dataset_path}/data/train-hint_sep.jsonl --output=${dataset_path}/data/openr1_hint_sep
```

### Run training
Run with `uv`:
```
# Test
python <PATH_TO>/hint_rl/cc_scripts/openmath_rl.py --config <PATH_TO>/hint_rl/cc_scripts/openmath_questa_50_grpo.yaml

# Actual run with similar hyperparameters as QuestA
python <PATH_TO>/hint_rl/cc_scripts/openmath_rl.py --config <PATH_TO>/hint_rl/cc_scripts/openmath_questa_50_grpo_slurm.yaml
```

### Run evaluations
Run with `uv`:
```
python <PATH_TO>/hint_rl/cc_scripts/olympiad_bench_eval.py --config <PATH_TO>/hint_rl/cc_scripts/eval-questa_50.yaml

# Slurm
dat_file=<PATH_TO>/eval_configs.dat <PATH_TO>/hint_rl/cc_scripts/slurm/eval_questa.sh
```

### Tensorboard
```
Compute node $: tensorboard --logdir=. --host 0.0.0.0 --load_fast false

Local $: ssh -N -f -L localhost:6007:<node_name>:6006 <username>@vulcan.alliancecan.ca
```


### OLD SETUP WITH APPTAINER
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