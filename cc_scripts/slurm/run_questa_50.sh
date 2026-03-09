#!/bin/bash

cd /workspace/hint_rl
uv pip install --no-deps -e .

ray start --head --disable-usage-stats
python3 /workspace/hint_rl/cc_scripts/openmath_rl.py --config /workspace/hint_rl/cc_scripts/openmath_questa_50_grpo.yaml scheduler.type=ray