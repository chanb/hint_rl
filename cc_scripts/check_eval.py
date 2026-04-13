import os

log_dir = "/home/chanb/scratch/hint_rl_results/logs/chanb/eval-math"
match_prefix = "local_eval-hint_rl"
for filename in os.listdir(log_dir):
    if not filename.startswith(match_prefix):
        continue

    rollout_dir = os.path.join(log_dir, filename, "rollout/0")

    if not os.path.isdir(rollout_dir):
        print(filename, "hasn't started")
        continue

    if len(os.listdir(rollout_dir)) != 1853:
        print(filename, "unfinished")
        continue
