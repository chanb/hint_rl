import argparse
import copy
import json
import multiprocessing
import numpy as np
import os

from areal.utils.pytest_util import run_test
from datasets import load_from_disk, load_dataset
from typing import Dict
from tqdm import tqdm


def check_correctness(sample, generation, timeout, debug=True):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""
    def _temp_run(sample, generation, debug, result):
        result.append(run_test(sample, test=generation, debug=debug))

    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(target=_temp_run, args=(sample, generation, debug, result))
    p.start()
    p.join()
    if p.is_alive():
        p.kill()
    if not result:
        in_outs = sample["test_cases"]
        # consider that all tests failed
        result = [[-1 for i in range(len(in_outs["inputs"]))]]
        if debug:
            print(f"global timeout")
    return result[0]


def split_prefix(text, scale):
    length = len(text)
    length *= scale
    pre_text = text[:int(length)]
    suf_text = text[int(length):]
    return pre_text, suf_text


def main():
    # conver to dict_keys(['query_id', 'verify', 'prompt', 'final_answer', 'answer'])
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='~/scratch/datasets/opencode/train.jsonl')
    parser.add_argument("--out_path", type=str, default='~/scratch/datasets/opencode/train-prefix.jsonl')
    args = parser.parse_args()

    ds = load_dataset("open-r1/OpenThoughts-114k-Code_decontaminated", split="train")
    for sample_i, sample in tqdm(enumerate(ds)):
        if not sample["source"] not in ["codeforces", "code_contests"]:
            continue

        if not sample["test_cases"]:
            continue

        if sample["num_tokens"] > 24000:
            continue

        deepseek_solution = sample["deepseek_solution"]
        hint_solution_split = deepseek_solution.split("### Solution Code")
        hint = hint_solution_split[0]
        accepted_solution = (
            hint_solution_split[1]
            .split("### Explanation")[0]
            .split("```python")[1]
            .split("```")[0]
        )

        test_cases = sample["test_cases"]
        sample["test_cases"] = json.loads(sample["test_cases"])

        if not all(check_correctness(sample, accepted_solution, TIMEOUT, debug=False)):
            continue

        if hint[0] == "\"":
            hint = hint[1:]
            hint = hint[:-1]
        prefix, suffix = split_prefix(hint, 1)
        if len(prefix) < 10:
            prefix = ""

        count += 1
        new_d = {}
        new_d['query_id'] = sample_i
        new_d['verify'] = True
        new_d['prompt'] = sample["problem"]
        new_d['hint'] = prefix
        new_d['task'] = 'code'
        new_d['solutions'] = ['```python\n'+accepted_solution+'```']
        new_d['test_cases'] = test_cases
        with open(args.out_path, "a") as A:
            A.write(json.dumps((new_d), ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
