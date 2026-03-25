import argparse
from datasets import load_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_input", type=str, default=None, required=True)
    parser.add_argument("--test_input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None, required=True)
    args = parser.parse_args()

    data_files = {
        "train": args.train_input,
    }

    if args.test_input is not None:
        data_files["test"] = args.test_input

    ds = load_dataset(
        "json",
        data_files=data_files,
    )

    print(ds)

    ds.save_to_disk(args.output)
