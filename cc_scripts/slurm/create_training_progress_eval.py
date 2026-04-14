import argparse
import os


def main(args):
    models_dir = args.models_dir
    dataset_path = args.dataset_path
    exp_name = args.exp_name

    assert os.path.isdir(models_dir)

    checkpoint_dirs = sorted(
        [filename for filename in os.listdir(models_dir) if "globalstep" in filename],
        key=lambda key: int(key.split("globalstep")[-1]),
    )

    dat_content = ""
    for checkpoint_name in checkpoint_dirs:
        print(checkpoint_name)

        trial_name = f"local_eval-{exp_name}-{checkpoint_name}"
        actor_path = os.path.join(models_dir, checkpoint_name)

        dat_content += f"export trial_name={trial_name} actor_path={actor_path} dataset_path={dataset_path}"
        dat_content += "\n"

    print(f"{len(checkpoint_dirs)} models to evaluate")

    with open(f"eval_configs-train_curve-{exp_name}.dat", "w+") as f:
        f.write(dat_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=str, help="The directory containing trained models")
    parser.add_argument("--dataset_path", type=str, help="The dataset to use")
    parser.add_argument("--exp_name", type=str, help="The experiment name")
    args = parser.parse_args()

    main(args)
