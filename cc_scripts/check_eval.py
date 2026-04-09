import os

log_dir = "/home/chanb/scratch/hint_rl_results/logs/chanb/openmath-dapo"
match_prefix = "local_eval-hint_rl_163-epoch"
for filename in os.listdir(log_dir):
    if not filename.startswith(match_prefix):
        continue

    rollout_dir = os.path.join(log_dir, filename, "rollout/0")

    if len(os.listdir(rollout_dir)) != 1853:
        print(filename)
