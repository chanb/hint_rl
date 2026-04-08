from datasets import load_from_disk, load_dataset

from testing_util import run_test
import json, os
import multiprocessing
import numpy as np
from typing import Dict

TIMEOUT = 10


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
        in_outs = sample["public_tests"]
        # consider that all tests failed
        result = [[-1 for i in range(len(in_outs["input"]))]]
        if debug:
            print(f"global timeout")
    return result[0]

# ── Load dataset ──────────────────────────────────────────────────────────────
# ds = load_from_disk("/home/chanb/scratch/datasets/taco/data/taco_hint_sep/train")
ds = load_dataset("open-r1/codeforces-cots", "solutions_py_decontaminated", split="train")

"""
dict_keys(['id', 'aliases', 'contest_id', 'contest_name', 'contest_type', 'contest_start', 'contest_start_year', 'index', 'time_limit', 'memory_limit', 'title', 'description', 'input_format', 'output_format', 'interaction_format', 'note', 'examples', 'editorial', 'prompt', 'generation', 'finish_reason', 'api_metadata', 'messages', 'accepted_solutions', 'failed_solutions', 'generated_tests', 'private_tests', 'problem_type', 'public_tests', 'public_tests_ms'])

{'input': ['5\n3 10 8 6 11\n4\n1\n10\n3\n11\n'], 'output': ['0\n4\n1\n5\n']}
"""

count = 0
for sample_i, sample in enumerate(ds):
    if not sample["accepted_solutions"]:
        continue

    print(sample_i)
    print(sample.keys())

    print(sample["accepted_solutions"])

    print(check_correctness(sample, sample["accepted_solutions"][0]["code"], TIMEOUT, debug=False))

    count += 1
    if count > 10:
        break
