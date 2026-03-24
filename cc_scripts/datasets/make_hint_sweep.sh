#!/bin/bash
source /home/chanb/research/hint_rl/hint_rl/.venv/bin/activate

python add_prefix.py --data_path=/home/chanb/scratch/datasets/questa/OpenR1-50-0-4.jsonl --out_path=/home/chanb/scratch/datasets/questa/data/OpenR1-0-0-4-prefix.jsonl --ratio=0
python process.py --input=/home/chanb/scratch/datasets/questa/data/OpenR1-0-0-4-prefix.jsonl --output=/home/chanb/scratch/datasets/questa/data/train-0-0-4.jsonl
python convert2hf.py --train_input=/home/chanb/scratch/datasets/questa/data/train-0-0-4.jsonl --output=/home/chanb/scratch/datasets/questa/data/openr1_0

python add_prefix.py --data_path=/home/chanb/scratch/datasets/questa/OpenR1-50-0-4.jsonl --out_path=/home/chanb/scratch/datasets/questa/data/OpenR1-10-0-4-prefix.jsonl --ratio=10
python process.py --input=/home/chanb/scratch/datasets/questa/data/OpenR1-10-0-4-prefix.jsonl --output=/home/chanb/scratch/datasets/questa/data/train-10-0-4.jsonl
python convert2hf.py --train_input=/home/chanb/scratch/datasets/questa/data/train-10-0-4.jsonl --output=/home/chanb/scratch/datasets/questa/data/openr1_10

python add_prefix.py --data_path=/home/chanb/scratch/datasets/questa/OpenR1-50-0-4.jsonl --out_path=/home/chanb/scratch/datasets/questa/data/OpenR1-20-0-4-prefix.jsonl --ratio=20
python process.py --input=/home/chanb/scratch/datasets/questa/data/OpenR1-20-0-4-prefix.jsonl --output=/home/chanb/scratch/datasets/questa/data/train-20-0-4.jsonl
python convert2hf.py --train_input=/home/chanb/scratch/datasets/questa/data/train-20-0-4.jsonl --output=/home/chanb/scratch/datasets/questa/data/openr1_20

python add_prefix.py --data_path=/home/chanb/scratch/datasets/questa/OpenR1-50-0-4.jsonl --out_path=/home/chanb/scratch/datasets/questa/data/OpenR1-30-0-4-prefix.jsonl --ratio=30
python process.py --input=/home/chanb/scratch/datasets/questa/data/OpenR1-30-0-4-prefix.jsonl --output=/home/chanb/scratch/datasets/questa/data/train-30-0-4.jsonl
python convert2hf.py --train_input=/home/chanb/scratch/datasets/questa/data/train-30-0-4.jsonl --output=/home/chanb/scratch/datasets/questa/data/openr1_30

python add_prefix.py --data_path=/home/chanb/scratch/datasets/questa/OpenR1-50-0-4.jsonl --out_path=/home/chanb/scratch/datasets/questa/data/OpenR1-40-0-4-prefix.jsonl --ratio=40
python process.py --input=/home/chanb/scratch/datasets/questa/data/OpenR1-40-0-4-prefix.jsonl --output=/home/chanb/scratch/datasets/questa/data/train-40-0-4.jsonl
python convert2hf.py --train_input=/home/chanb/scratch/datasets/questa/data/train-40-0-4.jsonl --output=/home/chanb/scratch/datasets/questa/data/openr1_40

python add_prefix.py --data_path=/home/chanb/scratch/datasets/questa/OpenR1-50-0-4.jsonl --out_path=/home/chanb/scratch/datasets/questa/data/OpenR1-50-0-4-prefix.jsonl --ratio=50
python process.py --input=/home/chanb/scratch/datasets/questa/data/OpenR1-50-0-4-prefix.jsonl --output=/home/chanb/scratch/datasets/questa/data/train-50-0-4.jsonl
python convert2hf.py --train_input=/home/chanb/scratch/datasets/questa/data/train-50-0-4.jsonl --output=/home/chanb/scratch/datasets/questa/data/openr1_50

python add_prefix.py --data_path=/home/chanb/scratch/datasets/questa/OpenR1-50-0-4.jsonl --out_path=/home/chanb/scratch/datasets/questa/data/OpenR1-60-0-4-prefix.jsonl --ratio=60
python process.py --input=/home/chanb/scratch/datasets/questa/data/OpenR1-60-0-4-prefix.jsonl --output=/home/chanb/scratch/datasets/questa/data/train-60-0-4.jsonl
python convert2hf.py --train_input=/home/chanb/scratch/datasets/questa/data/train-60-0-4.jsonl --output=/home/chanb/scratch/datasets/questa/data/openr1_60

python add_prefix.py --data_path=/home/chanb/scratch/datasets/questa/OpenR1-50-0-4.jsonl --out_path=/home/chanb/scratch/datasets/questa/data/OpenR1-70-0-4-prefix.jsonl --ratio=70
python process.py --input=/home/chanb/scratch/datasets/questa/data/OpenR1-70-0-4-prefix.jsonl --output=/home/chanb/scratch/datasets/questa/data/train-70-0-4.jsonl
python convert2hf.py --train_input=/home/chanb/scratch/datasets/questa/data/train-70-0-4.jsonl --output=/home/chanb/scratch/datasets/questa/data/openr1_70

python add_prefix.py --data_path=/home/chanb/scratch/datasets/questa/OpenR1-50-0-4.jsonl --out_path=/home/chanb/scratch/datasets/questa/data/OpenR1-80-0-4-prefix.jsonl --ratio=80
python process.py --input=/home/chanb/scratch/datasets/questa/data/OpenR1-80-0-4-prefix.jsonl --output=/home/chanb/scratch/datasets/questa/data/train-80-0-4.jsonl
python convert2hf.py --train_input=/home/chanb/scratch/datasets/questa/data/train-80-0-4.jsonl --output=/home/chanb/scratch/datasets/questa/data/openr1_80

python add_prefix.py --data_path=/home/chanb/scratch/datasets/questa/OpenR1-50-0-4.jsonl --out_path=/home/chanb/scratch/datasets/questa/data/OpenR1-90-0-4-prefix.jsonl --ratio=90
python process.py --input=/home/chanb/scratch/datasets/questa/data/OpenR1-90-0-4-prefix.jsonl --output=/home/chanb/scratch/datasets/questa/data/train-90-0-4.jsonl
python convert2hf.py --train_input=/home/chanb/scratch/datasets/questa/data/train-90-0-4.jsonl --output=/home/chanb/scratch/datasets/questa/data/openr1_90

python add_prefix.py --data_path=/home/chanb/scratch/datasets/questa/OpenR1-50-0-4.jsonl --out_path=/home/chanb/scratch/datasets/questa/data/OpenR1-100-0-4-prefix.jsonl --ratio=100
python process.py --input=/home/chanb/scratch/datasets/questa/data/OpenR1-100-0-4-prefix.jsonl --output=/home/chanb/scratch/datasets/questa/data/train-100-0-4.jsonl
python convert2hf.py --train_input=/home/chanb/scratch/datasets/questa/data/train-100-0-4.jsonl --output=/home/chanb/scratch/datasets/questa/data/openr1_100