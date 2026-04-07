import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default='~/scratch/datasets/questa/OpenR1-50-0-4-prefix.jsonl')
    parser.add_argument("--output", type=str, default='~/scratch/datasets/questa/data/train.jsonl')
    args = parser.parse_args()


    with open(args.input, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            new_data = {}
            new_data['question'] = data['prompt']
            new_data['answer'] = data['solutions']
            new_data['id'] = data['query_id']

            if 'hint' in data:
                new_data['hint'] = data['hint']

            if 'test_cases' in data:
                new_data['test_cases'] = data['test_cases']

            if 'starter_code' in data:
                new_data['starter_code'] = data['starter_code']

            with open(args.output, 'a', encoding='utf-8') as f:
                f.write(json.dumps(new_data, ensure_ascii=False) + '\n')
