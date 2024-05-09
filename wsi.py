#import init
import polylm
import util
from options import Options
from data import Vocabulary
from train import get_multisense_vocab


def main():
    options = Options()
    options.model_dir = "models/"
    options.corpus_path = "data/processed_corpus.txt"
    options.vocab_path = "data/corpus_vocab.txt"
    
    vocab = Vocabulary(options.vocab_path, options.min_occurrences_for_vocab, build=False)
    multisense_vocab = get_multisense_vocab(options.n_senses_file, vocab, options)
    model = polylm.PolyLM(
            vocab, options, multisense_vocab=multisense_vocab, training=False)

    util.wsi(model, vocab, options)


if __name__ == "__main__":
    main()
