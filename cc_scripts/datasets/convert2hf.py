from datasets import load_dataset

if __name__ == "__main__":
    ds = load_dataset(
        "json",
        data_files={
            "train": "/home/chanb/scratch/datasets/questa/data/train.jsonl",
            "test": "/home/chanb/scratch/datasets/questa/data/test.jsonl"
        }
    )

    print(ds)

    ds.save_to_disk("/home/chanb/scratch/datasets/questa/data/openr1_50")
