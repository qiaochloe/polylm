import polylm
from data import Vocabulary
from data import Corpus
from options import Options


import csv

def get_multisense_vocab(path, vocab, options):
    if path:
        n_senses = {}
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                token = parts[0]
                n = int(parts[1])
                vocab_id = vocab.str2id(token)
                if vocab_id == vocab.unk_vocab_id:
                    logging.warn('Token "%s" not in vocabulary.' % token)
                #else:
                n_senses[vocab_id] = n
    else:
        n_senses = {
                t: options.max_senses_per_word
                for t in range(vocab.size)
                if vocab.get_n_occurrences(t) >=
                        options.min_occurrences_for_polysemy
        }
        
    return n_senses


if __name__ == "__main__":
    options = Options()
    options.model_dir = "models/"
    options.corpus_path = "data/processed_corpus.txt"
    options.vocab_path = "data/corpus_vocab.txt"
    
    vocab = Vocabulary(options.vocab_path, options.min_occurrences_for_vocab, build=False)
    multisense_vocab = get_multisense_vocab(options.n_senses_file, vocab, options)
    
    model = polylm.PolyLM(vocab, options, multisense_vocab, training=True)
    corpus = Corpus(options.corpus_path, vocab)
    
    losses = model.train_model(corpus, 1)

    with open('losses.csv', 'w') as f:
        wr = csv.writer(f)
        wr.writerow(losses)
