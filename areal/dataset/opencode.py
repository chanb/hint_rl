from datasets import load_dataset, load_from_disk


def get_opencode_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    dataset = load_from_disk(path)[split]

    def process(sample):
        messages = [
            {
                "role": "user",
                "content": sample["question"]
                + "\nPlease reason step by step, and put your final answer within ```python\n...\n```.",
            }
        ]
        return {"messages": messages}

    dataset = dataset.map(process).remove_columns(["question"])

    # Filter out sequences longer than max_length if tokenizer and max_length are provided
    if max_length is not None:

        def filter_length(sample):
            # Tokenize the user content to check length
            content = sample["messages"][0]["content"]
            tokens = tokenizer.encode(content)
            return len(tokens) <= max_length

        dataset = dataset.filter(filter_length)

    return dataset
