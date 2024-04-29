import init
import math
import logging
import os
import sys
import time

import torch.nn
import torch
import numpy as np

import bert
import util

def masked_softmax(logits, mask):
    masked_logits = logits - 1e30 * (1.0 - mask)
    soft_max_f  = torch.nn.softmax(dim = -1)
    return soft_max_f(masked_logits)


class PolyLMModel(torch.nn.Module):
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
            # tf.nn.embedding_lookup
            self._qd = torch.nn.functional.embedding(self._target_positions, self._qd)

        else:
            pass
            #self._disambiguated_reps = self._get_single_sense_embeddings(
            #        self._masked_seqs)
        self._output_reps = self._prediction_layer(self._disambiguated_reps)
        
        flattened_reps = torch.reshape(
                self._output_reps, [-1, self._embedding_size])
        
        # (n_targets, embedding_size)
        self._target_reps = torch.nn.functional.embedding(self._target_positions, flattened_reps)
        # (n_targets, total_senses)
        target_position_scores = (
                torch.matmul(self._target_reps,
                          torch.transpose(self._embeddings, 0, 1)) +
                torch.unsqueeze(self._biases_with_dummy, 0) -
                1e30 * torch.unsqueeze(self._unpredictable_tokens, 0))
        torch_softmax = torch.nn.softmax(dim = 1)
        target_position_probs = torch_softmax(target_position_scores)
        target_sense_indices = torch.nn.functional.embedding(self._targets,
                self._sense_indices)
        self.target_sense_probs = self.manually_batched_gather(
                target_position_probs,
                target_sense_indices)
        target_sense_masks = torch.nn.functional.embedding(
                self._targets, self._sense_mask)
        self.target_sense_probs = self.target_sense_probs * target_sense_masks
        self.target_token_probs = torch.sum(self.target_sense_probs, axis=1)
        self.target_token_probs = torch.maximum(self.target_token_probs, 1e-30)
        log_target_probs = torch.log(self.target_token_probs)
        self.lm_loss = -torch.mean(log_target_probs)
        
        self._qp = self.target_sense_probs / torch.unsqueeze(
                self.target_token_probs, 1)

        targets_are_multisense = torch.nn.functional.embedding(self._targets,
                self._is_multisense)
        n_multisense = torch.sum(targets_are_multisense) + 1e-6

        sharpened_q = torch.pow(self._qp, self._dl_r)
        log_sharpened_q = torch.log(torch.sum(sharpened_q, axis=1))
        log_sharpened_q = log_sharpened_q * targets_are_multisense
        self.d_loss = -torch.sum(
                log_sharpened_q) / (self._dl_r * n_multisense)

        if self._has_disambiguation_layer:
            qp = self._qp.detach()
            qd = self._qd

            p_norms = torch.norm(qp, dim=1)
            d_norms = torch.norm(qd, dim=1)
            num = torch.sum(qp * qd, dim=1)
            div = p_norms * d_norms + 1e-10
            cosine_sim = num / div
            cosine_sim = cosine_sim * targets_are_multisense
            self.m_loss = -self._ml_coeff * torch.sum(
                    cosine_sim) / n_multisense

            self._update_mean_qd = self._update_smoothed_mean(
                    self._mean_qd, qd,
                    indices=torch.unsqueeze(self._targets, 1))
        else:
            self.m_loss = 0.0
            self._update_mean_qd = torch.nn.Identity()

        self._update_mean_qp = self._update_smoothed_mean(
                self._mean_qp, qp,
                indices=torch.unsqueeze(self._targets, 1))

        self.loss = self.lm_loss + self.d_loss + self.m_loss

        self._add_find_neighbours()
        self._add_get_mean_q()

    def manually_batched_gather(params, indices, axis):
        batch_dims=1
        result = []
        for p,i in zip(params, indices):
                r = torch.gather(input = p, dim = axis-batch_dims, index = i)
                result.append(r)
        return torch.stack(result)

    def _get_sense_embeddings(self, tokens):
        sense_indices = torch.nn.functional.embedding(tokens,
                self._sense_indices)
        return torch.nn.functional.embedding(sense_indices,
                self._embeddings)

    def _get_sense_embeddings_and_biases(self, tokens):
        sense_indices = torch.nn.functional.embedding(tokens,
                self._sense_indices)
        sense_embeddings = torch.nn.functional.embedding(sense_indices,
                self._embeddings)
        sense_biases = torch.nn.functional.embedding(sense_indices,
                self._biases_with_dummy)
        return sense_embeddings, sense_biases

    def _make_word_embeddings(self, seqs, sense_weights=None):
        # ids.shape + (n_senses, embedding_size)
        sense_embeddings = self._get_sense_embeddings(seqs)
        
        if sense_weights is None:
            # ids.shape + (n_senses,)
            sense_weight_logits = torch.nn.functional.embedding(seqs,
                    self._sense_weight_logits)
            sense_mask = torch.nn.functional.embedding(seqs,
                    self._sense_mask)
            sense_weights = masked_softmax(
                    sense_weight_logits, sense_mask)

        # ids.shape + (embedding_size,)
        return torch.squeeze(torch.matmul(
                torch.transpose(sense_embeddings,0,1),
                torch.unsqueeze(sense_weights, dim=-1)), dim = -1)

    def _calculate_sense_probs(self, seqs, reps):
        # ids.shape + (n_senses, embedding_size)
        sense_embeddings, sense_biases = self._get_sense_embeddings_and_biases(
                seqs)
        # ids.shape + (n_senses, 1)
        sense_scores = torch.matmul(
                sense_embeddings, torch.unsqueeze(reps, dim=-1))
        # ids.shape + (n_senses)
        sense_scores = torch.squeeze(sense_scores, dim = -1)
        sense_scores = sense_scores + sense_biases
        sense_mask = torch.nn.functional.embedding(seqs,
                self._sense_mask)
        return masked_softmax(sense_scores, sense_mask) 
    
#for layers we will likely need to convert to classes --> and then check Bert implementation to 
# walk over it. 
    def _disambiguation_layer(self, seqs):
        word_embeddings = self._make_word_embeddings(seqs)

        reps = bert.BertModel(self._disambiguation_bert_config)(word_embeddings, self._padding)

        # (batch_size, sentence_len, embedding_size)
        #reps = model.get_output()
        # (batch_size, sentence_len, n_senses)
        sense_probs = self._calculate_sense_probs(seqs, reps)
        
        # (batch_size, sentence_len, embedding_size)
        disambiguated_reps = self._make_word_embeddings(
                seqs, sense_weights=sense_probs)

        return disambiguated_reps, sense_probs

    def _prediction_layer(self, reps):
        return  bert.BertModel(self._disambiguation_bert_config)(reps, self._padding)
            
    #change how to 
    def _update_smoothed_mean(self, mean, values, indices=None, weight=0.005):
        if indices is None:
            #need to deal with tf.assign. lol 
            mean = (1.0 - weight) * mean + weight * values
            return mean

        current_values = torch.gather(mean, indices)
        updates = weight * (values - current_values)
        
        return mean.scatter_add_(0, indices, updates)

    def add_to_feed_dict(self, feed_dict, batch, dl_r, ml_coeff):
        padding = np.zeros(batch.unmasked_seqs.shape, dtype=np.int32)
        for i, l in enumerate(batch.seq_len):
            padding[i, :l] = 1
        feed_dict.update({
                self._unmasked_seqs: batch.unmasked_seqs,
                self._masked_seqs: batch.masked_seqs,
                self._padding: padding,
                self._target_positions: batch.target_positions,
                self._targets: batch.targets,
                self._dl_r: dl_r,
                self._ml_coeff: ml_coeff})

    def contextualize(self, sess, batch):
        padding = np.zeros(batch.masked_seqs.shape, dtype=np.int32)
        for i, l in enumerate(batch.seq_len):
            padding[i, :l] = 1
        feed_dict = {
                self._masked_seqs: batch.masked_seqs,
                self._padding: padding,
                self._target_positions: batch.target_positions}
        return sess.run(self._target_reps, feed_dict=feed_dict)

    def disambiguate(self, sess, batch, method='prediction'):
        padding = np.zeros(batch.masked_seqs.shape, dtype=np.int32)
        for i, l in enumerate(batch.seq_len):
            padding[i, :l] = 1
        feed_dict = {
                self._unmasked_seqs: batch.unmasked_seqs,
                self._masked_seqs: batch.masked_seqs,
                self._padding: padding,
                self._target_positions: batch.target_positions,
                self._targets: batch.targets}
        prob_tensors = {'prediction': self._qp}
        if self._has_disambiguation_layer:
            prob_tensors['disambiguation'] = self._qd
        return sess.run(prob_tensors[method], feed_dict=feed_dict)

    def get_target_probs(self, sess, batch):
        padding = np.zeros(batch.masked_seqs.shape, dtype=np.int32)
        for i, l in enumerate(batch.seq_len):
            padding[i, :l] = 1
        feed_dict = {
                self._unmasked_seqs: batch.unmasked_seqs,
                self._masked_seqs: batch.masked_seqs,
                self._padding: padding,
                self._target_positions: batch.target_positions,
                self._targets: batch.targets}
        return sess.run([self.target_token_probs, self.target_sense_probs],
                        feed_dict=feed_dict)

    def _add_get_mean_q(self):
        self._mean_q_tokens =  torch.tensor([None], dtype=torch.int32)
        self._selected_mean_qp = torch.nn.functional.embedding(self._mean_q_tokens,
                self._mean_qp)
        self._selected_mean_qd = torch.nn.functional.embedding(self._mean_q_tokens,
                self._mean_qd)

    def _get_mean_sense_probs(self, sess, tokens):
        request = {
                'qp': self._selected_mean_qp,
                'qd': self._selected_mean_qd,
        }
        feed_dict = {self._mean_q_tokens: tokens}
        return sess.run(request, feed_dict=feed_dict)

    def get_sense_embeddings(self, sess):
        return sess.run(self._embeddings)

    def _add_find_neighbours(self):
        #torch.tensor([None, None], dtype=torch.int32)


        self._interesting_ids = torch.tensor([None], dtype=torch.int32)
        self._n_neighbours = torch.tensor([], dtype=torch.int32)

        sense_indices = torch.nn.functional.embedding(self._interesting_ids,
                self._sense_indices) 
        interesting_embeddings =  torch.nn.functional.embedding(sense_indices,
                self._embeddings)
        interesting_embeddings = torch.reshape(
                interesting_embeddings, [-1, self._embedding_size])
        interesting_norms = torch.norm(
                interesting_embeddings, dim=1)

        norms = torch.norm(self._embeddings, dim=1)

        # (n_interesting, vocab.size*n_senses)
        dot = torch.matmul(interesting_embeddings,
                        torch.transpose(self._embeddings,0,1))
        dot = dot / torch.unsqueeze(interesting_norms, dim=1)
        dot = dot / torch.unsqueeze(norms, dim=0)
        cosine_similarities = torch.reshape(
                dot,
                [-1, self._max_senses, self._total_senses])
        mask = torch.concat([
                torch.tensor([2.0]),
                torch.zeros([self._total_senses - 1], dtype=torch.float32)],
                dim=0)
        mask = torch.unsqueeze(mask, dim=0)
        mask = torch.unsqueeze(mask, dim=0)
        cosine_similarities = cosine_similarities - mask

        self._neighbour_similarities, indices = torch.top_k(
                cosine_similarities, k=self._n_neighbours)
        self._neighbour_similarities = self._neighbour_similarities[:, :, 1:]
        self._neighbour_tokens = torch.nn.functional.embedding(indices,
                self._sense_to_token)[:, :, 1:]
        self._neighbour_sense_nums = torch.nn.functional.embedding(indices,
                self._sense_to_sense_num)[:, :, 1:]

    def get_neighbours(self, sess, tokens, n=10):
        feed_dict = {self._interesting_ids: tokens,
                     self._n_neighbours: n}
        similarities, tokens, sense_nums = sess.run(
                [self._neighbour_similarities,
                 self._neighbour_tokens,
                 self._neighbour_sense_nums],
                feed_dict=feed_dict)
        return similarities, tokens, sense_nums


def _deduplicated_indexed_slices(values, indices):
    # Calculate unique indices and their counts
    unique_indices, inverse_indices = torch.unique(indices, return_inverse=True)
    counts = torch.bincount(inverse_indices)

    # Calculate the summed values using scatter_add
    summed_values = torch.zeros_like(unique_indices, dtype=values.dtype)
    summed_values.scatter_add_(0, inverse_indices.unsqueeze(0).expand_as(values), values)

    return summed_values, unique_indices

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        g0, v0 = grad_and_vars[0]

        if g0 is None:
            average_grads.append((g0, v0))
            continue

        if isinstance(g0, torch.sparse_coo_tensor):
            indices = []
            values = []
            for g, v in grad_and_vars:
                indices.append(g.indices)
                values.append(g.values)
            all_indices = torch.cat(indices, dim=0)
            avg_values = torch.cat(values, dim=0) / len(grad_and_vars)
            av, ai = _deduplicated_indexed_slices(avg_values, all_indices)
            grad = torch.sparse_coo_tensor(av, ai, g0.size())
        else:
            grads = []
            for g, _ in grad_and_vars:
                expanded_g = torch.unsqueeze(g, 0)
                grads.append(expanded_g)

            grad = torch.cat(grads, dim=0)
            grad = torch.mean(grad, dim=0)

        average_grads.append((grad, v0))

    return average_grads

def clip_gradients(grads_and_vars, val):
    grads = [g for g, v in grads_and_vars]
    var = [v for g, v in grads_and_vars]
    clipped_grads, grad_norm = torch.nn.utils.clip_grad_norm_ (grads, val)
    return list(zip(clipped_grads, var)), grad_norm


class PolyLM(torch.nn.Module):

    def __init__(self, vocab, options,
                 multisense_vocab={}, training=False):
        self._vocab = vocab
        self._options = options
        self._max_seq_len = self._options.max_seq_len
        self._embedding_size = self._options.embedding_size
        self._max_senses = self._options.max_senses_per_word

        gpus = [int(x) for x in self._options.gpus.split(',')]
        logging.info(
                'Building PolyLM on GPU(s) ' +
                ', '.join([str(x) for x in gpus]))
        self._n_towers = len(gpus)

        #self._global_step = tf.get_variable(
        #        'global_step', [],
        #        initializer=tf.zeros_initializer(),
        #        dtype=tf.int32,
        #        trainable=False) 
        
        self._global_step = torch.nn.Parameter(torch.tensor(0), dtype=torch.int32, requires_grad=False)
        self._learning_rate = torch.tensor([], dtype=torch.int32)
        self._opt = torch.nn.optim.Adam(self.parameters(), lr=self._learning_rate.item())

        self._n_senses = np.ones([self._vocab.size], dtype=np.int32)
        for t, n in multisense_vocab.items():
            assert n > 0 and n <= self._max_senses
            self._n_senses[t] = n

        self._towers = []
        self._grads = []
        self._losses = []
        self._lm_losses = []
        self._d_losses = []
        self._m_losses = []

        for i in range(self._n_towers):
            #with torch.device('/gpu:%d' % i):
                #will need to change this as it has variable scope into another class system. 
                #with tf.variable_scope('polylm', reuse=tf.AUTO_REUSE):
                tower = PolyLMModel(
                        self._vocab, self._n_senses,
                        self._options, training=training)
                self._towers.append(tower)
                self._losses.append(tower.loss)
                self._grads.append(
                        self._opt.compute_gradients(tower.loss))
                self._lm_losses.append(tower.lm_loss)
                self._d_losses.append(tower.d_loss)
                self._m_losses.append(tower.m_loss)

        self._default_model = self._towers[0]
        self._loss = torch.mean(torch.stack(self._losses))
        self._lm_loss = torch.mean(torch.stack(self._lm_losses))
        self._d_loss = torch.mean(torch.stack(self._d_losses))
        self._m_loss = torch.mean(torch.stack(self._m_losses))
        
        grads_and_vars = average_gradients(self._grads)
        clipped_grads, self._grad_norm = clip_gradients(
                grads_and_vars, self._options.max_gradient_norm)
        self._update_params = self._opt.apply_gradients(
                clipped_grads, global_step=self._global_step)
        
        #I don't know if this works at all
        self._update = self.update_all

        self._saver = torch.save(self._default_model.state_dict(), 'checkpoint.pth')

    
    def update_all(self):
        self._update_params,
        self._default_model._update_mean_qp
        self._default_model._update_mean_qd

    def attempt_restore(self, sess, model_dir, expect_exists):
        if self._options.checkpoint_version != -1:
            path = '%s/polylm.ckpt-%d' % (
                    model_dir, self._options.checkpoint_version)
        else:
            logging.info('Looking for model at %s...' % model_dir)
            #path = tf.train.latest_checkpoint(model_dir)
            path =  torch.load(model_dir)

        if os.path.exists(path):
            #sess.run(tf.global_variables_initializer())
            self.model.load_state_dict(torch.load(path))
            logging.info('Reading model parameters from %s' % path)
            #self._saver.restore(sess, path)
        else:
            if expect_exists:
                bad_path = path if path else model_dir
                raise Exception('There is no saved checkpoint at %s' %
                                bad_path)
            else:
                logging.info('There is no saved checkpoint at %s. '
                             'Creating model with fresh parameters.' %
                             model_dir)
                #sess.run(tf.global_variables_initializer())
                #otherwise we just make a new one right?
                self.__init__

        #n_params = sum(v.get_shape().num_elements()
         #              for v in tf.trainable_variables()
         #              if v.get_shape().num_elements() is not None)
        #logging.info('Num params: %d' % n_params)

    def get_embeddings(self, sess):
        return self._default_model.get_embeddings(sess)

    def get_mask(self, sess):
        return self._default_model.get_mask(sess)

    def _train_on_batch(self, sess, batches, step_num):
        assert len(batches) == self._n_towers
        if step_num < self._options.lr_warmup_steps:
            lr_ratio = (step_num + 1) / self._options.lr_warmup_steps
        elif self._options.anneal_lr:
            lr_ratio = (self._options.n_batches -
                        step_num) / self._options.n_batches
        else:
            lr_ratio = 1.0
        learning_rate = lr_ratio * self._options.learning_rate

        if step_num < self._options.dl_warmup_steps:
            dl_ratio = step_num / self._options.dl_warmup_steps
            dl_r = 1.0 + dl_ratio * (self._options.dl_r - 1.0)
        else:
            dl_r = self._options.dl_r

        if step_num < self._options.ml_warmup_steps:
            ml_ratio = step_num / self._options.ml_warmup_steps
            ml_coeff = ml_ratio * self._options.ml_coeff
        else:
            ml_coeff = self._options.ml_coeff

        feed_dict = {self._learning_rate: learning_rate}
        for i, batch in enumerate(batches):
            self._towers[i].add_to_feed_dict(
                    feed_dict, batch, dl_r, ml_coeff)

        fetches = {
                'update': self._update(),
                'loss': self._loss,
                'lm_loss': self._lm_loss,
                'd_loss': self._d_loss,
                'm_loss': self._m_loss,
                'global_step': self._global_step,
                'grad_norm': self._grad_norm
        }

        start_time = time.time()
        batch_output = sess.run(fetches, feed_dict=feed_dict)
        end_time = time.time()

        batch_output['time'] = end_time - start_time
        batch_output['lr'] = learning_rate
        batch_output['dl_r'] = dl_r
        batch_output['ml_coeff'] = ml_coeff

        return batch_output

    def get_n_senses(self, vocab_id):
        return self._n_senses[vocab_id]

    def get_sense_probs(self, sess, vocab_ids):
        return self._default_model._get_mean_sense_probs(sess, vocab_ids)[0]

    def train(self, corpus, sess, test_words=[]):
        logging.info('Commencing training...')
        logging.info('Vocab size: %d' % self._vocab.size)
        logging.info('Total senses: %d' % np.sum(self._n_senses))

        checkpoint_path = os.path.join(self._options.model_dir, 'polylm.ckpt')
        test_tokens = [self._vocab.str2id(w) for w in test_words]
        
        self.display_words(sess, test_words)
        
        gpu_time_for_block = 0.0
        loss_for_block = 0.0
        lm_loss_for_block = 0.0
        d_loss_for_block = 0.0
        m_loss_for_block = 0.0
        norm_for_block = 0.0
        #global_step = 1
        tokens_read = 0
        masking_policy = [
                float(x) for x in self._options.masking_policy.split()]
        logging.info('Masking policy: %.2f [MASK], %.2f self, %.2f random' % (
                masking_policy[0],
                masking_policy[1],
                masking_policy[2]))
        batch_gen = corpus.generate_batches(
                self._options.batch_size, self._options.max_seq_len,
                self._options.mask_prob, variable_length=True,
                masking_policy=masking_policy)
        batch_num = sess.run(self._global_step)
        block_start_batch = batch_num
        block_start_time = time.time()
        while batch_num < self._options.n_batches:
            batch_num += 1
            batches = []
            for i in range(self._n_towers):
                batch = next(batch_gen)
                batches.append(batch)
                tokens_read += batch.n_tokens()

            batch_output = self._train_on_batch(sess, batches, batch_num)

            gpu_time_for_block += batch_output['time']
            loss_for_block += batch_output['loss']
            lm_loss_for_block += batch_output['lm_loss']
            d_loss_for_block += batch_output['d_loss']
            m_loss_for_block += batch_output['m_loss']
            norm_for_block += batch_output['grad_norm']

            if batch_num % self._options.print_every == 0:
                block_end_time = time.time()
                block_size = batch_num - block_start_batch
                time_for_block = block_end_time - block_start_time
                logging.info(
                        'Batch %d, %d tokens: gpu time = %.3fs, '
                        'total time = %.3fs, lm_loss = %.5f, d_loss = %.5f, '
                        'm_loss = %.5f, grad norm = %.5f, loss = %.5f' % (
                                batch_num, tokens_read,
                                gpu_time_for_block, time_for_block,
                                lm_loss_for_block / block_size,
                                d_loss_for_block / block_size,
                                m_loss_for_block / block_size,
                                norm_for_block / block_size,
                                loss_for_block / block_size))
                gpu_time_for_block = 0.0
                loss_for_block = 0.0
                lm_loss_for_block = 0.0
                d_loss_for_block = 0.0
                m_loss_for_block = 0.0
                norm_for_block = 0.0
                block_start_batch = batch_num
                block_start_time = block_end_time

            if batch_num % self._options.test_every == 0:
                self.display_words(sess, test_words)
                logging.info(
                        'lr = %.8f, dl_r = %.5f, ml_coeff = %.5f' % (
                                batch_output['lr'],
                                batch_output['dl_r'],
                                batch_output['ml_coeff']))

            if batch_num % self._options.save_every == 0:
                logging.info('Saving to %s...' % checkpoint_path)
                self._saver.save(sess, checkpoint_path,
                                 global_step=self._global_step)


    def disambiguate(self, sess, batch, method='prediction'):
        return self._default_model.disambiguate(
                sess, batch, method=method)

    def contextualize(self, sess, batch):
        return self._default_model.contextualize(sess, batch)

    def get_target_probs(self, sess, batch):
        return self._default_model.get_target_probs(sess, batch)

    def match(self, sess, keys, values):
        return self._default_model.match(sess, keys, values)

    def get_top_k_substitutes(self, sess, reps, k):
        return self._default_model.get_top_k_substitutes(
                sess, reps, k)

    def display_words(self, sess, words):
        vocab_ids = [self._vocab.str2id(t) for t in words]
        words = [self._vocab.id2str(i) for i in vocab_ids]
        sense_stats = self._default_model._get_mean_sense_probs(
                sess, vocab_ids)
        similarities, tokens, senses = self._default_model.get_neighbours(
                sess, vocab_ids)
        
        for i, (vocab_id, word) in enumerate(zip(vocab_ids, words)):
            info = [
                    'qd=%.4f, qp=%.4f' % (
                            sense_stats['qd'][i, s],
                            sense_stats['qp'][i, s])
                    for s in range(self._n_senses[vocab_id])]
            util.display_word(self._vocab, word, similarities[i, :, :],
                              tokens[i, :, :], senses[i, :, :],
                              self._n_senses[vocab_id], info=info)


        