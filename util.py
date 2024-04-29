import logging

def display_word(corpus, word, similarities, tokens, senses, n_senses,
                 info=None):
    
    vocab_id = corpus.str2id(word)
    for sense in range(n_senses):
        neighbour_strs = [
                '%s_%d(%.3f)' % (corpus.id2str(t), s+1, c)
                for c, t, s in zip(similarities[sense, :],
                                   tokens[sense, :],
                                   senses[sense, :])]
        info_str = '' if info is None else ' (%s)' % info[sense]
        logging.info('%s_%d%s: %s' % (
                word, sense + 1, info_str, ' '.join(neighbour_strs)))