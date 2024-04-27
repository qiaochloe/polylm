# polylm

# Setting up repository

Set up your virtual environment.
```bash
python3.10 -m venv .venv 
```

Activate your environment
```bash
.\.venv\scripts\activate # Windows
source .venv/bin/activate # MacOS
```

Install PyTorch using the instructions on this [site](https://pytorch.org/get-started/locally/). Choose the stable, pip, python, and default installation.

Install other packages. 
```bash
pip install -r requirements.txt
```

## Training

For training, download the bookcorpus dataset and place it in the `data/` folder. Then run the `preprocessing.py` script. 

The following command can be used to train a model with the same parameters as PolyLM<sub>BASE</sub>:

    python train.py --model_dir=models/ --corpus_path=data/bookcorpus/books_large_p1.txt --vocab_path=data/bookcorpus/vocab.txt --embedding_size=256 --bert_intermediate_size=1024 --n_disambiguation_layers=4 --n_prediction_layers=12 --max_senses_per_word=8 --min_occurrences_for_vocab=500 --min_occurrences_for_polysemy=20000 --max_seq_len=128 --gpus=0 --batch_size=32 --n_batches=6000000 --dl_warmup_steps=2000000 --ml_warmup_steps=1000000 --dl_r=1.5 --ml_coeff=0.1 --learning_rate=0.00003 --print_every=100 --save_every=10000

## Testing
It is possible to use the download scripts provided in the `models` folder.

    cd models
    ./download-lemmatized-large.sh

First download the SemEval 2010 and 2013 WSI datasets:

    cd data
    ./download-wsi.sh
    cd ..

Activate NLTK's WordNet capabilities:

    python -c "import nltk; nltk.download('wordnet')"

Download [Stanford CoreNLP's part-of-speech tagger v3.9.2](https://nlp.stanford.edu/software/stanford-postagger-2018-10-16.zip) and put the folder in the root. It is required to perform lemmatization when evaluating on WSI.

PolyLM evaluation can be performed as follows:

    ./wsi.sh data/wsi/SemEval-2010 SemEval-2010 /models/polylm-lemmatized-large --gpus 0 --pos_tagger_root /stanford-postagger-2018-10-16/models/english-bidirectional-distsim.tagger

    ./wsi.sh data/wsi/SemEval-2013 SemEval-2013 /models/polylm-lemmatized-large --gpus 0 --pos_tagger_root /stanford-postagger-2018-10-16/models/english-bidirectional-distsim.tagger


Note that inference is only supported on a single GPU currently, but is generally very fast.

