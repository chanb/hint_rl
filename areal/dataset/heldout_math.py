from datasets import load_dataset

SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def get_aime25_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    dataset = load_dataset(path="math-ai/aime25", split="test")

    def process(sample):
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": sample["problem"]
            }
        ]
        return {"messages": messages}

    dataset = dataset.map(process).remove_columns(["problem"])

    # Filter out sequences longer than max_length if tokenizer and max_length are provided
    if max_length is not None:

        def filter_length(sample):
            # Tokenize the user content to check length
            content = sample["messages"][1]["content"]
            tokens = tokenizer.encode(content)
            return len(tokens) <= max_length

        dataset = dataset.filter(filter_length)

    return dataset


def get_aime24_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    dataset = load_dataset(path="math-ai/aime24", split="test")

    def process(sample):
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": sample["problem"]
            }
        ]
        return {"messages": messages, "answer": sample["solution"]}

    dataset = dataset.map(process).remove_columns(["problem", "solution"])

    # Filter out sequences longer than max_length if tokenizer and max_length are provided
    if max_length is not None:

        def filter_length(sample):
            # Tokenize the user content to check length
            content = sample["messages"][1]["content"]
            tokens = tokenizer.encode(content)
            return len(tokens) <= max_length

        dataset = dataset.filter(filter_length)

    return dataset



def get_hmmt_feb_2025_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    dataset = load_dataset(path="MathArena/hmmt_feb_2025", split="train")

    def process(sample):
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": sample["problem"]
            }
        ]
        return {"messages": messages}

    dataset = dataset.map(process).remove_columns(["problem"])

    # Filter out sequences longer than max_length if tokenizer and max_length are provided
    if max_length is not None:

        def filter_length(sample):
            # Tokenize the user content to check length
            content = sample["messages"][1]["content"]
            tokens = tokenizer.encode(content)
            return len(tokens) <= max_length

        dataset = dataset.filter(filter_length)

    return dataset



def get_brumo_2025_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    dataset = load_dataset(path="MathArena/brumo_2025", split="train")

    def process(sample):
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": sample["problem"]
            }
        ]
        return {"messages": messages}

    dataset = dataset.map(process).remove_columns(["problem"])

    # Filter out sequences longer than max_length if tokenizer and max_length are provided
    if max_length is not None:

        def filter_length(sample):
            # Tokenize the user content to check length
            content = sample["messages"][1]["content"]
            tokens = tokenizer.encode(content)
            return len(tokens) <= max_length

        dataset = dataset.filter(filter_length)

    return dataset
