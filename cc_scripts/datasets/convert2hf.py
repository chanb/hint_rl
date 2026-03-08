from datasets import load_dataset

if __name__ == "__main__":
    ds = load_dataset(
        "json",
        data_files={
            "train": "/datasets/questa/data/train.jsonl",
            "test": "/datasets/questa/data/test.jsonl"
        }
    )

    print(ds)

    ds.save_to_disk("/datasets/questa/openr1_50")