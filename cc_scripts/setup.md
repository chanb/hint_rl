## First time setup
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

## Datasets
### QuestA datasets
The `jsonl` files are stored in [here](https://huggingface.co/datasets/foreverlasting1202/QuestA/tree/main).
The `make_hint_sweep.sh` file will create datasets with hints from 0% to 100%, with 10% increments, stored in `<PATH_TO>/datasets/questa/data/`:
```
mkdir -p <PATH_TO>/datasets/questa/data/
cd <PATH_TO>/datasets/questa
wget https://huggingface.co/datasets/foreverlasting1202/QuestA/resolve/main/OpenR1-50-0-4.jsonl

mkdir -p <PATH_TO>/datasets/apps/data/
cd <PATH_TO>/datasets/apps
wget https://huggingface.co/datasets/codeparrot/apps/resolve/main/train.jsonl

cd <PATH_TO>/hint_rl/cc_scripts/datasets
code_path=<PATH_TO>/hint_rl dataset_path=<PATH_TO>/datasets/questa ./make_hint_sweep.sh
```

### Hint RL dataset
Rather than fixing a particular %, we store the hints in a separate column and dynamically provide hints in training:
```
cd <PATH_TO>/hint_rl/cc_scripts/datasets

# Math: Open Math
export dataset_path=<PATH_TO>/datasets/questa
mkdir -p <PATH_TO>/datasets/questa/data
python create_hint_dataset_openmath.py --data_path=${dataset_path}/OpenR1-50-0-4.jsonl --out_path=${dataset_path}/data/OpenR1-hint_sep.jsonl
python process.py --input=${dataset_path}/data/OpenR1-hint_sep.jsonl --output=${dataset_path}/data/train-hint_sep.jsonl
python convert2hf.py --train_input=${dataset_path}/data/train-hint_sep.jsonl --output=${dataset_path}/data/openr1_hint_sep

# Code: Open code
export dataset_path=<PATH_TO>/datasets/opencode
mkdir -p <PATH_TO>/datasets/opencode/data
python create_hint_dataset_opencode.py --out_path=${dataset_path}/data/opencode-hint_sep.jsonl
python process.py --input=${dataset_path}/data/opencode-hint_sep.jsonl --output=${dataset_path}/data/train-hint_sep.jsonl
python convert2hf.py --train_input=${dataset_path}/data/train-hint_sep.jsonl --output=${dataset_path}/data/opencode_hint_sep
```

For open code we do extra filtering which evaluates nvidia/OpenReasoning-Nemotron-1.5B on the dataset
```
sbatch eval_code-with_hints.sh
# TODO
```

## Training
Run with `uv`:
```
# Locally with GPUs, for example to train on QuestA math dataset:
python <PATH_TO>/hint_rl/cc_scripts/openmath_rl.py --config <PATH_TO>/hint_rl/cc_scripts/configs/train/openmath_hint_rl.yaml

# Generally, to run with slurm:
sbatch <PATH_TO>/hint_rl/cc_scripts/slurm/train_*.sh
```

### Tuning guide
AReaL provides a [documentation](https://www.inclusion-ai.org/AReaL/en/best_practices/handling_oom.html) on tuning the hyperparameters for memory usage.
Based on our tuning, `max_concurrent_rollouts` and `allocation_mode` are the most impactful.
On L40s nodes we can have approximately 28 `max_concurent_rollouts` per GPU.
The sweet spot on Vulcan seems to be two nodes for rollout and two nodes for training, e.g., `sglang:d2p1t1+fsdp:d2p1t1`.


### Hint RL visualization
Run `cc_scripts/plots/check_hint_change.ipynb` to plot out how the hint % changes over time.
Run `cc_scripts/plots/check_hint_usefulness-trained.ipynb` to plot out the learning curve of the trained model---this requires running evaluation on some datasets.

### Tensorboard
```
Compute node $: tensorboard --logdir=. --host 0.0.0.0 --load_fast false

Local $: ssh -N -f -L localhost:6007:<node_name>:6006 <username>@vulcan.alliancecan.ca
```


## Evaluations
Run with `uv`:
```
python <PATH_TO>/hint_rl/cc_scripts/eval_math.py --config <PATH_TO>/hint_rl/cc_scripts/configs/eval/eval_math.yaml

# Slurm
dat_file=<PATH_TO>/eval_configs-*.dat <PATH_TO>/hint_rl/cc_scripts/slurm/eval_*.sh
```

## Experiments
### Math domain
```
cd <PATH_TO>/hint_rl/cc_scripts

# Ours
sbatch slurm/train_hint_rl.sh

# Baseline 1: QuestA, 50% hint for 100 steps -> 25% hint for remaining steps
sbatch slurm/train_questa.sh

# Baseline 2: DAPO, 0% hint
sbatch slurm/train_dapo.sh

# Baseline 3: On-policy self-distillation, distilling from X% hint
sbatch slurm/train_opsd.sh
```

### Code domain
TBD


## Ablation experiments in math domain
### Hint percentage
```
dat_file=<PATH_TO>/eval_configs-per_hint_percentage.dat <PATH_TO>/hint_rl/cc_scripts/slurm/eval_math.sh
```
Plot with `cc_scripts/plots/check_hint_usefulness-per_hint_percentage.ipynb`

### Hint usefulness per pretrained model
```
dat_file=<PATH_TO>/eval_configs-per_model.dat <PATH_TO>/hint_rl/cc_scripts/slurm/eval_math.sh
```
Plot with `cc_scripts/plots/check_hint_usefulness-per_model.ipynb`

## Hint RL code changes
To account for the new datasets, the changes are done in `areal.dataset`.

We implement dynamic hints by adding `areal.workflow.dynamic_hint_rlvr.DynamicHintRLVRWorkflow` and `areal.trainer.rl_trainer.CurriculumPPOTrainer`.
The former adds partial hints based on `hint_percentage` of the question, and the latter keeps track of the `hint_percentage`.

To include code domains, we added `CodeVerifyWorker` under `areal.reward`, as well as `areal.utils.pyext2` and `areal.utils.pytest_util` which we imported from the [TACO repository](https://github.com/FlagOpen/TACO).
