import json
import argparse

from datasets import load_dataset
from tqdm import tqdm

def split_prefix(text, scale):
    length = len(text)
    length *= scale
    pre_text = text[:int(length)]
    suf_text = text[int(length):]
    return pre_text, suf_text


def main():
    # conver to dict_keys(['query_id', 'verify', 'prompt', 'final_answer', 'answer'])
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='~/scratch/datasets/apps/train.jsonl')
    parser.add_argument("--out_path", type=str, default='~/scratch/datasets/apps/train-prefix.jsonl')
    args = parser.parse_args()

    ds = load_dataset("nvidia/OpenCodeReasoning", "split_1")["split_1"]
    with open(args.data_path, "r") as f:
        lines = f.readlines()
    
    count = 0
    hit_idxes = []
    for i, item in tqdm(enumerate(ds)):
        if item["dataset"] != "apps":
            continue

        if item["split"] != "train":
            continue

        if int(item["index"]) >= len(lines):
            print(f"Index {item['index']} out of range for lines with length {len(lines)}")
            continue

        if int(item["index"]) in hit_idxes:
            continue

        line = lines[int(item["index"])]
        d = json.loads(line)

        text = item['output']
        if text[0] == "\"":
            text = text[1:]
            text = text[:-1]
        solution = text.split('</think>')[-1]
        prefix, suffix = split_prefix(solution, 1)
        if len(prefix) < 10:
            prefix = ""
        answer = text
        final_answer = item['solution']
        if final_answer not in solution:
            print("skipped", i)
            continue

        hit_idxes.append(int(item["index"]))
        count += 1
        new_d = {}
        new_d['query_id'] = item["index"]
        new_d['verify'] = True
        new_d['prompt'] = d["question"]
        new_d['hint'] = prefix
        new_d['task'] = 'code'
        new_d['solutions'] = ['```python\n'+final_answer+'```']
        new_d['test_cases'] = d["input_output"]
        with open(args.out_path, "a") as A:
            A.write(json.dumps((new_d), ensure_ascii=False) + '\n')
    print("Wrote {} samples.".format(count))
    print(max(hit_idxes), min(hit_idxes))


if __name__ == '__main__':
    main()
