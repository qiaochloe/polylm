import torch.nn
import torch
import numpy as np
import bert

class PolyLM(torch.nn.Module):
    def __init__(self, vocab, n_senses, options, training=False):
        self._vocab = vocab
        self._embedding_size = options.embedding_size
        self._max_seq_len = options.max_seq_len
        self._max_senses = options.max_senses_per_word
        self._training = training
        self._has_disambiguation_layer = (
                options.use_disambiguation_layer and self._max_senses > 1)
        
        self._disambiguation_bert_config = bert.BertConfig(
                hidden_size=self._embedding_size,
                num_hidden_layers=options.n_disambiguation_layers,
                intermediate_size=options.bert_intermediate_size,
                num_attention_heads=options.n_attention_heads,
                hidden_dropout_prob=options.dropout,
                attention_probs_dropout_prob=options.dropout,
                max_position_embeddings=self._max_seq_len)

        self._prediction_bert_config = bert.BertConfig(
                hidden_size=self._embedding_size,
                num_hidden_layers=options.n_prediction_layers,
                intermediate_size=options.bert_intermediate_size,
                num_attention_heads=options.n_attention_heads,
                hidden_dropout_prob=options.dropout,
                attention_probs_dropout_prob=options.dropout,
                max_position_embeddings=self._max_seq_len)

        self._unmasked_seqs = torch.tensor([None, None], dtype=torch.int32)
        self._masked_seqs = torch.tensor([None, None], dtype=torch.int32)

        self._padding = torch.tensor([None, None], dtype=torch.int32)
        self._targets = torch.tensor([None], dtype=torch.int32)

        self._target_positions = torch.tensor([None], dtype = torch.int32)

        self._dl_r = torch.tensor([], dtype = torch.float32)
        self._ml_coeff = torch.tensor([], dtype = torch.float32)

        self._total_senses = np.sum(n_senses) + 1
        sense_indices = np.zeros(
                [self._vocab.size, self._max_senses], dtype=np.int32)
        sense_to_token = np.zeros(
                [self._total_senses], dtype=np.int32)
        sense_to_sense_num = np.zeros(
                [self._total_senses], dtype=np.int32)
        sense_mask = np.zeros(
                [self._vocab.size, self._max_senses], dtype=np.float32)
        is_multisense = np.zeros([self._vocab.size], dtype=np.float32)
        index = 1

        for i, n in enumerate(n_senses):
            if n > 1:
                is_multisense[i] = 1.0
            for j in range(n):
                sense_indices[i, j] = index
                sense_mask[i, j] = 1.0
                sense_to_token[index] = i
                sense_to_sense_num[index] = j
                index += 1

        self._sense_indices = sense_indices
        self._sense_mask = sense_mask
        self._is_multisense = is_multisense
        self._n_senses = n_senses
        self._sense_to_token = sense_to_token
        self._sense_to_sense_num = sense_to_sense_num

        self._embedding_initializer = torch.nn.Embedding(self._total_senses, self._embedding_size)
        bias_zeros = np.zeros(self._total_senses - 1)
        dummy = np.concatenate([-1e30], bias_zeros)
        self.biases = torch.tensor(dummy)
        self.sense_weight_logits = np.zeros_like(sense_mask)
        no_predict_tokens = [
                self._vocab.bos_vocab_id, self._vocab.eos_vocab_id,
                self._vocab.pad_vocab_id, self._vocab.mask_vocab_id]
        unpredictable_tokens = np.zeros(
                [self._total_senses], dtype=np.float32)
        for t in no_predict_tokens:
            unpredictable_tokens[sense_indices[t, 0]] = 1.0
        self._unpredictable_tokens = torch.tensor(
                unpredictable_tokens)
        mean_prob = sense_mask / np.sum(
                sense_mask, axis=1, keepdims=True)
        
        self._mean_qp = np.full_like(sense_mask, mean_prob)

        self._mean_qd = np.full_like(sense_mask, mean_prob)

        if options.use_disambiguation_layer:
            self._disambiguated_reps, _ = self._disambiguation_layer(
                    self._masked_seqs)
            _, self._qd = self._disambiguation_layer(
                    self._unmasked_seqs)
            self._qd = torch.reshape(self._qd, [-1, self._max_senses])
            self._qd = tf.nn.embedding_lookup(
                    self._qd, self._target_positions)

        