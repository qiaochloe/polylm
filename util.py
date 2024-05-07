import numpy as np
import os
import itertools
import sys
import logging

import data

from xml.etree import ElementTree
from nltk.tokenize.stanford import StanfordTokenizer
from nltk.tag.stanford import StanfordTagger
from nltk.stem.wordnet import WordNetLemmatizer

_POS_TAGGER_ROOT = '.'

np.set_printoptions(linewidth=np.inf)

PLURAL = '<PL>'
COMP = '<COMP>'
SUP = '<SUP>'
PAST = '<PAST>'
GER = '<GER>'
NONTHIRD = '<N3RD>'
THIRD = '<3RD>'
PART = '<PART>'

APPEND = {
        'NNS': PLURAL,
        'JJR': COMP,
        'JJS': SUP,
        'RBR': COMP,
        'RBS': SUP,
        'VBD': PAST,
        'VBG': GER,
        'VBP': NONTHIRD,
        'VBZ': THIRD,
        'VBN': PART
}

MAP = {
        'NNS': 'n',
        'JJR': 'a',
        'JJS': 'a',
        'RBR': 'r',
        'RBS': 'r',
        'VBD': 'v',
        'VBG': 'v',
        'VBP': 'v',
        'VBZ': 'v',
        'VBN': 'v'
}

def stem_focus(focus, pos):
    if focus == 'lay':
        return 'lie', PAST
    stemmed = lemmatize(focus, pos)
    if focus.endswith('ing'):
        suf = GER
    elif focus.endswith('s'):
        if pos == 'v':
            suf = THIRD
        else:
            suf = PLURAL
    elif focus.endswith('ed'):
        suf = PAST
    else:
        suf = PART
    return stemmed, suf

def stem(sentences):
    pos_tagger_root = _POS_TAGGER_ROOT
    tagger = StanfordPOSTagger(
            pos_tagger_root + '/models/english-left3words-distsim.tagger',
            path_to_jar=pos_tagger_root + '/stanford-postagger.jar')
    for s in sentences:
        s.append('\n')
    joined = list(itertools.chain.from_iterable(sentences))
    logging.info('About to dispatch to Stanford POS tagger')
    output = tagger.tag(joined)
    logging.info('Got result from Stanford POS tagger')
    #print(output)
    start = 0
    stemmed_sentences = []
    alignments = []
    tags = []
    for s in sentences:
        #logging.info(s)
        end = start + len(s) - 1
        #logging.info(output[start:end])
        tagged_sentence = output[start:end]
        stemmed_sentence = []
        alignment = []
        tag = []
        for token, pos in tagged_sentence:
            token = token.lower()
            alignment.append(len(stemmed_sentence))
            if pos in MAP and token.isalpha():
                stemmed_token = lemmatize(token, MAP[pos])
                stemmed_sentence.append(stemmed_token)
                stemmed_sentence.append(APPEND[pos])
            else:
                stemmed_sentence.append(token)
            tag.append(pos)

        stemmed_sentences.append(stemmed_sentence)
        alignments.append(alignment)
        tags.append(tag)
        start = end

    assert start == len(output)
    return stemmed_sentences, alignments, tags

class StanfordPOSTagger(StanfordTagger):

    _SEPARATOR = '_'
    _JAR = 'stanford-postagger.jar'

    def __init__(self, *args, **kwargs):
        super(StanfordPOSTagger, self).__init__(*args, **kwargs)

    @property
    def _cmd(self):
        return [
                'edu.stanford.nlp.tagger.maxent.MaxentTagger',
                '-model',
                self._stanford_model,
                '-textFile',
                self._input_file_path,
                '-tokenize',
                'false',
                '-outputFormatOptions',
                'keepEmptySentences',
                '-sentenceDelimiter',
                'newline',
                '-options',
                _TOKENIZER_OPTIONS_STR
        ]

class Tokenizer(StanfordTokenizer):
    def tokenize(self, s):
        cmd = ['edu.stanford.nlp.process.PTBTokenizer', '-preserveLines']
        return self._parse_tokenized_output(self._execute(cmd, s))

_TOKEN_MAP = {"``": '"', "''": '"', '--': '-', "/?": "?", "/.": ".", '-LRB-': '(', '-RRB-': ')'}
_TOKENIZER_OPTIONS = {'tokenizePerLine': 'true',
                      'americanize': 'false',
                      'normalizeCurrency': 'false',
                      'normalizeParentheses': 'false',
                      'normalizeOtherBrackets': 'false',
                      'asciiQuotes': 'false',
                      'latexQuotes': 'false',
                      'unicodeQuotes': 'false',
                      'ptb3Ellipsis': 'false',
                      'unicodeEllipsis': 'false',
                      'ptb3Dashes': 'false',
                      'splitHyphenated': 'true'}
_TOKENIZER_OPTIONS_STR = ','.join(
        ['%s=%s' % item for item in _TOKENIZER_OPTIONS.items()])
_LEMMATIZER = WordNetLemmatizer()

def lemmatize(word, pos):
    return _LEMMATIZER.lemmatize(word, pos)

def map_token(token):
    return _TOKEN_MAP.get(token, token)

def preprocess_str(s):
    return ' '.join([map_token(t) for t in s.split()])

def tokenize(seqs, tokenizer):
    tokenized_seqs = tokenizer.tokenize('\n'.join(seqs))
    tokenized_seqs = [s.split() for s in tokenized_seqs]
    alignments = []
    for seq, tok in zip(seqs, tokenized_seqs):
        alignment = []
        i = 0
        build = ''
        for word in seq.split():
            while True:
                build += tok[i]
                i += 1
                if build == word:
                    alignment.append(i-1)
                    build = ''
                    break
                if i >= len(tok):
                    logging.error('Bad tokenization:\n%s\n%s' %
                            (seq, ' '.join(tok)))
                    break
                    #sys.exit(0)
            if i >= len(tok):
                break
        alignments.append(alignment)
    tokenized_seqs = [[map_token(t) for t in s] for s in tokenized_seqs]
    return tokenized_seqs, alignments

def get_tokenizer():
    return Tokenizer(
            path_to_jar=_POS_TAGGER_ROOT + '/stanford-postagger.jar',
            options=_TOKENIZER_OPTIONS)

class WsiCorpus(object):

    def __init__(self, examples, vocab, stem=False):
        self._vocab = vocab
        self._lemmas = []
        self._pos = []
        self._instance_names = []
        self._sentences = []
        seqs = []

        for inst_name, lemma, pos, before, target, after in examples:
            before = preprocess_str(before)
            after = preprocess_str(after)
            self._instance_names.append(inst_name)
            self._lemmas.append(lemma)
            self._pos.append(pos)
            self._sentences.append(' '.join([before, target, after]))
            seqs.append(before)
            seqs.append(target)
            seqs.append(after)

        tokens, _ = tokenize(seqs, get_tokenizer())
        self._focuses = []
        self._tokens = []
        for i in range(0, len(tokens), 3):
            before_tokens, target_tokens, after_tokens = tokens[i:i + 3]
            self._focuses.append(len(before_tokens))
            self._tokens.append(before_tokens + target_tokens + after_tokens)

        if stem:
            self.stem()

    def stem(self):
        logging.info('Stemming WSI corpus')
        stemmed, alignments, tags = stem(self._tokens)
        logging.info('Finished stemming corpus')
        for i, alignment in enumerate(alignments):
            self._focuses[i] = alignment[self._focuses[i]]
            toks = stemmed[i]
            ind = self._focuses[i]
            stemmed_focus = toks[ind]
            if stemmed_focus != self._lemmas[i]:
                new_stem, suf = stem_focus(stemmed_focus, self._pos[i])
                if new_stem == self._lemmas[i]:
                    stemmed[i] = toks[:ind] + [new_stem, suf] + toks[ind+1:]
                    #logging.warn('fixed %s -> %s' % (' '.join(toks), ' '.join(stemmed[i])))
                else:
                    logging.warn('%s: stemmed focus "%s" does not match lemma "%s" in sentence "%s"' % (
                            self._instance_names[i], stemmed_focus, self._lemmas[i], ' '.join(stemmed[i])))
        self._tokens = []
        for seq in stemmed:
            self._tokens.append(seq)

    def generate_instances(self):
        for tokens, focus, lemma, pos, name, sentence in zip(
                self._tokens, self._focuses, self._lemmas,
                self._pos, self._instance_names, self._sentences):
            yield {'tokens': tokens, 'index_in_seq': focus, 'name': name,
                   'lemma': lemma, 'pos': pos, 'sentence': sentence}

    def _generate_batches(self, max_batch_size, max_seq_len):
        instance_gen = self.generate_instances()
        done = False
        while not done:
            instances = []
            while len(instances) < max_batch_size:
                try:
                    instance = next(instance_gen)
                    instances.append(instance)
                except StopIteration:
                    done = True
                    break
            yield instances, make_batch(instances, self._vocab, max_seq_len)

    def calculate_sense_probs(
            self, sess, model, max_batch_size, max_seq_length, method='prediction'):
        all_instances = []
        seen = set()
        for instances, batch in self._generate_batches(
                max_batch_size, max_seq_length):
            sense_probs = model.disambiguate(sess, batch, method=method)
            for i, instance in enumerate(instances):
                instance['sense_probs'] = sense_probs[i, :]
                all_instances.append(instance)

        return all_instances

def make_span(vocab_ids, index, vocab, max_seq_length):
    before = (max_seq_length - 2) // 2
    after = max_seq_length - 2 - before - 1
    left_space = index
    right_space = len(vocab_ids) - index - 1
    extra_left = max(0, after - right_space)
    extra_right = max(0, before - left_space)
    start = max(0, index - before - extra_left)
    end = min(len(vocab_ids), index + after + extra_right + 1)
    assert end - start + 2 <= max_seq_length
    span = [vocab.bos_vocab_id] + vocab_ids[start:end] + [vocab.eos_vocab_id]
    index_in_span = index - start + 1
    return span, index_in_span

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

def to_sentence(vocab, vocab_ids, highlight=None):
    def to_str(vocab_id, index):
        word = vocab.id2str(vocab_id)
        if highlight is not None and index == highlight:
            word = '**%s**' % word
        return word

    return ' '.join([to_str(t, i) for i, t in enumerate(vocab_ids)])

def make_batch(instances, vocab, max_seq_len,
               mask=True, replace_with_lemma=True):
    n_instances = len(instances)
    ids = vocab.pad_vocab_id * np.ones(
            [n_instances, max_seq_len], dtype=np.int32)
    seq_len = np.zeros([n_instances], dtype=np.int32)
    masked_indices = np.zeros([n_instances, 2], dtype=np.int32)
    masked_ids = np.zeros([n_instances], dtype=np.int32)
    unmasked_seqs = vocab.pad_vocab_id * np.ones(
            [n_instances, max_seq_len], dtype=np.int32)
    masked_seqs = vocab.pad_vocab_id * np.ones(
            [n_instances, max_seq_len], dtype=np.int32)
    target_positions = np.zeros([n_instances], dtype=np.int32)
    targets = np.zeros([n_instances], dtype=np.int32)

    for i, instance in enumerate(instances):
        vocab_ids = [vocab.str2id(t) for t in instance['tokens']]
        span, index = make_span(
                vocab_ids, instance['index_in_seq'], vocab, max_seq_len)
        instance['span'] = span
        instance['index_in_span'] = index
        unmasked_seqs[i, :len(span)] = span
        masked_seqs[i, :len(span)] = span
        if mask:
            masked_seqs[i, index] = vocab.mask_vocab_id
        seq_len[i] = len(span)
        target_positions[i] = i * max_seq_len + index
        if replace_with_lemma:
            targets[i] = vocab.str2id(instance['lemma'])
        else:
            targets[i] = span[index]

    return data.Batch(
            unmasked_seqs, masked_seqs, seq_len, target_positions, targets)

def generate_sem_eval_wsi_2010(dir_path, vocab, stem=False):
    logging.info('reading SemEval dataset from %s' % dir_path)

    extra_mapping = {'lay': 'lie', 'figger': 'figure', 'figgered': 'figure', 'lah': 'lie',
                     'half-straightened': 'straighten'}

    examples = []
    for root_dir, dirs, files in os.walk(dir_path):  # "../paper-menuscript/resources/SemEval-2010/test_data/"):
        logging.info('In %s' % root_dir)
        #     path = root.split(os.sep)
        for file in files:
            if '.xml' in file:
                tree = ElementTree.parse(os.path.join(root_dir, file))
                root = tree.getroot()
                for child in root:
                    inst_name = child.tag
                    lemma = inst_name.split('.')[0]
                    pos = inst_name.split('.')[1]

                    #stemmed_lemma = basic_stem(lemma)

                    # pres_sent = child.text
                    #target_sent = child[0].text
                    before = ''
                    if child.text is not None:
                        before = child.text.strip()
                    
                    target_tokens = child[0].text.strip().split()
                    focus_index = -1
                    for i, token in enumerate(target_tokens):
                        token = token.lower()
                        token_lemma = lemmatize(token, pos)
                        if token_lemma == lemma or extra_mapping.get(token, '') == lemma:
                            if focus_index != -1:
                                #logging.warn('Duplicate occurrence of lemma "%s" in sentence "%s"' % (
                                #        lemma, child[0].text))
                                pass
                            else:
                                focus_index = i

                    if focus_index == -1:
                        logging.warn('Found no occurrences of lemma "%s" in sentence "%s"' % (
                                lemma, child[0].text))
                        continue

                    before += ' '.join(target_tokens[:focus_index])
                    target = target_tokens[focus_index]

                    after = ' '.join(target_tokens[focus_index+1:])
                    if child[0].tail is not None:
                        after += child[0].tail.strip()

                    #logging.info(' '.join(seq))
                    #logging.info(str(focus_index))

                    #corpus.add_example(seq, focus_index, lemma, pos, inst_name)
                    examples.append((inst_name, lemma, pos, before, target, after))

    return WsiCorpus(examples, vocab, stem=stem)

def generate_sem_eval_wsi_2013(path, vocab, stem=False):
    logging.info('reading SemEval dataset from %s' % path)

    examples = []
    tree = ElementTree.parse(path)
    root = tree.getroot()
    for lemma_element in root:
        lemma_name = lemma_element.get('item')
        logging.info(lemma_name)
        lemma = lemma_name.split('.')[0]
        pos = lemma_name.split('.')[1]

        for inst in lemma_element:
            inst_name = inst.get('id')
            context = inst[0]

            before = ''
            if context.text is not None:
                before = context.text.strip()

            target = context[0].text.strip()
            if len(target.split()) != 1:
                logging.warn(target)
                sys.exit()

            after = ''
            if context[0].tail is not None:
                after = context[0].tail.strip()

            examples.append((inst_name, lemma, pos, before, target, after))

    return WsiCorpus(examples, vocab, stem=stem)

def wsi(model, vocab, options):
    path = options.wsi_path
    if options.wsi_format == 'SemEval-2010':
        path = os.path.join(path, 'test_data')
        corpus = generate_sem_eval_wsi_2010(path, vocab, stem=options.lemmatize)
        allow_multiple = False
    elif options.wsi_format == 'SemEval-2013':
        path = os.path.join(
                path,
                'contexts/senseval2-format/'
                'semeval-2013-task-13-test-data.senseval2.xml')
        corpus = generate_sem_eval_wsi_2013(path, vocab, stem=options.lemmatize)
        allow_multiple = True
    else:
        logging.error('Unrecognized WSI format: "%s"' % options.wsi_format)
        return
    instances = corpus.calculate_sense_probs(
            model, options.batch_size, options.max_seq_len,
            method=options.sense_prob_source)
    lemmas = {}
    for instance in instances:
        lemmas.setdefault(instance['lemma'], []).append(instance)

    n_senses_count = np.zeros([options.max_senses_per_word + 1], dtype=np.int32)
    for lemma, instances in lemmas.items():
        for i, instance in enumerate(instances):
            lemma = instance['lemma']
            pos = instance['pos']
            sense_probs = instance['sense_probs']
            cluster_num = np.argmax(sense_probs)
            if allow_multiple:
                sense_labels = [
                        '%s.%s.%d/%.4f' % (lemma, pos, n+1, p)
                        for n, p in enumerate(sense_probs)
                        if p > options.wsi_2013_thresh or n == cluster_num]
                n_senses_count[len(sense_labels)] += 1
                score_str = ' '.join(sense_labels)
            else:
                score_str = '%s.%s.%d' % (lemma, pos, cluster_num+1)

            #logging.info(instance['sentence'])
            logging.info(to_sentence(vocab, instance['span'], instance['index_in_span']))
            logging.info('(%s)' % ('/'.join(['%.3f' % p for p in instance['sense_probs']])))
            print('%s.%s %s %s' % (lemma, pos, instance['name'], score_str))
        model.display_words([lemma])

    if allow_multiple:
        n_senses_prob = n_senses_count / np.sum(n_senses_count)
        for i, p in enumerate(n_senses_prob):
            logging.info('Probability of assigning %d senses: %.4f' % (i, p))
        mean = np.sum(np.arange(options.max_senses_per_word + 1) * n_senses_prob)
        logging.info('Mean number of senses: %.4f' % mean)