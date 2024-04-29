#!/bin/bash

#SBATCH -J polylm-tf
#SBATCH -N 1
#SBATCH --time=1:00:00
#SBATCH --mem=4G

#SBATCH -o polylm-tf-%j.out
#SBATCH -e polylm-tf-%j.out

python train.py --model_dir=models/ --corpus_path=data/bookcorpus/books_large_p1.txt --vocab_path=data/bookcorpus/vocab.txt --embedding_size=256 --bert_intermediate_size=1024 --n_disambiguation_layers=4 --n_prediction_layers=12 --max_senses_per_word=8 --min_occurrences_for_vocab=500 --min_occurrences_for_polysemy=20000 --max_seq_len=128 --gpus=0 --batch_size=32 --n_batches=6000000 --dl_warmup_steps=2000000 --ml_warmup_steps=1000000 --dl_r=1.5 --ml_coeff=0.1 --learning_rate=0.00003 --print_every=100 --save_every=10000