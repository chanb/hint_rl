import argparse
import os

def main(args):
    dat_file = args.dat_file
    log_dir = args.log_dir

    eval_map = dict()
    for line in open(dat_file):
        trial_name = line.split("trial_name=")[1].split(" ")[0]
        eval_map[trial_name] = line

    patch_dat_content = ""
    seen_trial_names = set()
    for trial_name in os.listdir(log_dir):
        if trial_name not in eval_map:
            continue

        seen_trial_names.add(trial_name)

        rollout_dir = os.path.join(log_dir, trial_name, "rollout/0")
        if not os.path.isdir(rollout_dir) or len(os.listdir(rollout_dir)) != 1853:
            print(trial_name)
            patch_dat_content += eval_map[trial_name]

    all_trial_names = set(eval_map.keys())
    new_trial_names = all_trial_names.difference(seen_trial_names)

    for trial_name in new_trial_names:
        print(trial_name)
        patch_dat_content += eval_map[trial_name]

    new_dat_file = dat_file.split("/")
    new_dat_file = "/".join(new_dat_file[:-1]) + "/" + new_dat_file[-1].split(".dat")[0] + "-patched.dat"

    with open(new_dat_file, "w+") as f:
        f.write(patch_dat_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dat_file", type=str, help="The dat file that specifies the evaluation models")
    parser.add_argument("--log_dir", type=str, help="The directory storing the evaluation results")
    args = parser.parse_args()

    main(args)
