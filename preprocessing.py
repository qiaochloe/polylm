from data import Vocabulary

corpus_path = "data/bookcorpus/books_large_p1.txt"
vocab_path = "data/bookcorpus/vocab.txt"

vocab = Vocabulary(corpus_path, min_occurrences=500, build=True)
vocab.write_vocab_file(vocab_path)