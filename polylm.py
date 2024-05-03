#import init
import math
import logging
import os
import sys
import time

import torch.nn
import torch
import numpy as np

from bert import BertModel, BertConfig
import util

def masked_softmax(logits, mask):
    masked_logits = logits - 1e30 * (1.0 - mask)
    return torch.nn.functional.softmax(masked_logits, dim=-1)

# model = PolyLMModel(vocab, n_senses, options, training=True)
# logits = model(input_ids, attention_mask, token_type_ids)
# loss = model.calculate_loss(logits, targets)

class PolyLMModel(torch.nn.Module):
    def __init__(self, vocab, n_senses, options, training=False):
        super(PolyLMModel, self).__init__()

        # Initialization
        self.vocab = vocab
        self.embedding_size = options.embedding_size
        self.max_seq_len = options.max_seq_len
        self.max_senses = options.max_senses_per_word
        self.training = training

        self.has_disambiguation_layer = (options.use_disambiguation_layer and self.max_senses > 1)
        
        # Set up BERT
        self.disambiguation_bert = BertModel(BertConfig(
            hidden_size=self.embedding_size,
            num_hidden_layers=options.n_disambiguation_layers,
            intermediate_size=options.bert_intermediate_size,
            num_attention_heads=options.n_attention_heads,
            hidden_dropout_prob=options.dropout,
            attention_probs_dropout_prob=options.dropout,
            max_position_embeddings=self.max_seq_len,
        ))

        self.prediction_bert = BertModel(BertConfig(
            hidden_size=self.embedding_size,
            num_hidden_layers=options.n_prediction_layers,
            intermediate_size=options.bert_intermediate_size,
            num_attention_heads=options.n_attention_heads,
            hidden_dropout_prob=options.dropout,
            attention_probs_dropout_prob=options.dropout,
            max_position_embeddings=self.max_seq_len,
        ))

        # Initialize sense indices and masks
        total_senses = np.sum(n_senses) + 1
        sense_indices = np.zeros([self.vocab.size, self.max_senses], dtype=np.int32)
        sense_mask = np.zeros([self.vocab.size, self.max_senses], dtype=np.float32)
        is_multisense = np.zeros([self.vocab.size], dtype=np.float32)
        #sense_to_token = np.zeros([self.total_senses], dtype=np.int32)
        #sense_to_sense_num = np.zeros([self.total_senses], dtype=np.int32)

        index = 1
        for i, num_senses in enumerate(n_senses):
            if num_senses > 1:
                is_multisense[i] = 1.0
            for j in range(num_senses):
                sense_indices[i, j] = index
                sense_mask[i, j] = 1.0
                #sense_to_token[index] = i
                #sense_to_sense_num[index] = j
                index += 1

        self.register_buffer('sense_indices', torch.from_numpy(sense_indices))
        self.register_buffer('sense_mask', torch.from_numpy(sense_mask))
        self.register_buffer('is_multisense', torch.from_numpy(is_multisense))
        #self.n_senses = torch.tensor(n_senses)
        #self.sense_to_token = torch.tensor(sense_to_token)
        #self.sense_to_sense_num = torch.tensor(sense_to_sense_num)

        # Embeddings and biases
        self.embeddings = torch.nn.Embedding(total_senses, self.embedding_size)
        self.biases = torch.nn.Parameter(torch.zeros(total_senses - 1))
        #self.embeddings = torch.nn.Parameter(torch.randn(self.total_senses, self.embedding_size) / np.sqrt(self.embedding_size))
        #self.biases = torch.nn.Parameter(torch.zeros(self.total_senses - 1))
        self.sense_weight_logits = torch.nn.Parameter(torch.zeros(self.vocab.size, self.max_senses))

    def forward(unmasked_seqs, masked_seqs, padding, target_positions, targets, dl_r, ml_coeff):
        if self.options.use_disambiguation_layer:
            disambiguated_reps, _ = self.disambiguation_bert(masked_seqs)
            _, qd = self.disambiguation_bert(unmasked_seqs)
            qd = qd.view(-1, self.max_senses)
            #qd = torch.reshape(self.qd, (-1, self.max_senses))
            qd = torch.nn.functional.embedding(target_positions, qd)
        else:
            pass
            #self.disambiguated_reps = self.get_single_sense_embeddings(
            #        self.masked_seqs)
        
        output_reps = self.prediction_bert(disambiguated_reps)
        flattened_reps = output_reps.view(-1, self.embedding_size)
        target_reps = flattened_reps[target_positions]
        #flattened_reps = torch.reshape(self.output_reps, [-1, self.embedding_size])
        #target_reps = torch.nn.functional.embedding(self.target_positions, flattened_reps)

        target_scores = torch.matmul(target_reps, self.embeddings.weight.transpose(0, 1))
        target_scores += self.biases.unsqueeze(0)
        target_probs = torch.nn.functional.softmax(target_scores, dim=1)

        lm_loss = torch.nn.functional.cross_entropy(target_scores, target_positions)

        # NOTE:  This is the more advanced version which uses biases_with_dummy and unpredictable_tokens

        #unpredictable_tokens = np.zeros([self.total_senses], dtype=np.float32)

        # no_predict_tokens = [
        #        self.vocab.bos_vocab_id, 
        #        self.vocab.eos_vocab_id,
        #        self.vocab.pad_vocab_id, 
        #        self.vocab.mask_vocab_id
        #]

        #for t in no_predict_tokens:
        #    unpredictable_tokens[sense_indices[t, 0]] = 1.0
        #self.unpredictable_tokens = torch.tensor(unpredictable_tokens)

        #self.biases_with_dummy = torch.cat([torch.tensor([-1e30]), self.biases])

        #target_scores = (
        #    torch.matmul(self.target_reps, self.embeddings.t()) +
        #    torch.unsqueeze(self.biases_with_dummy, 0) -
        #    1e30 * torch.unsqueeze(self.unpredictable_tokens, 0))
        
        # TODO: CHECK THIS LM LOSS
        #target_sense_indices = torch.nn.functional.embedding(self.targets, self.sense_indices)

        # self.target_sense_probs = target_position_probs.gather(1, target_sense_indices)

        #target_sense_masks = torch.nn.functional.embedding(
        #        self.targets, self.sense_mask)
        #self.target_sense_probs = self.target_sense_probs * target_sense_masks
        #self.target_token_probs = torch.sum(self.target_sense_probs, axis=1)
        #self.target_token_probs = torch.maximum(self.target_token_probs, 1e-30)
        #log_target_probs = torch.log(self.target_token_probs)

        #self.lm_loss = -torch.mean(log_target_probs)

        if self.has_disambiguation_layer:
            # Calculatae disambiguation loss
            qp = torch.nn.functional.softmax(target_scores, dim=1)
            # self.qp = self.target_sense_probs / torch.unsqueeze(self.target_token_probs, 1)
            sharpened_q = qp ** dl_r # sharpened_q = torch.pow(self.qp, self.dl_r)
            sharpened_q = torch.log(sharpened_q.sum(dim=1))

            # TODO: Check is_multisense
            targets_are_multisense = torch.nn.functional.embedding(targets, self.is_multisense)
            n_multisense = torch.sum(targets_are_multisense) + 1e-6

            sharpened_q *= targets_are_multisense
            d_loss = -torch.sum(log_sharpened_q) / (dl_r * n_multisense)

            # Calculate metric loss
            p_norms = torch.norm(qp, dim=1)
            d_norms = torch.norm(qd, dim=1)
            cosine_sim = torch.sum(qp * qd, dim=1) / (p_norms * d_norms + 1e-10)
            cosine_sim *= targets_are_multisense
            m_loss = -ml_coeff * torch.sum(cosine_sim) / n_multisense

        else:
            d_loss = torch.tensor(0.0) 
            m_loss = torch.tensor(0.0)
            
        total_loss = lm_loss + d_loss + m_loss

        # NOTE: ignore update_mean_qd

        return total_loss, lm_loss, d_loss, m_loss

                

        # self.embedding_initializer = torch.nn.Embedding(self.total_senses, self.embedding_size)
        
        # bias_zeros = np.zeros(self.total_senses - 1)
        # dummy = np.concatenate([-1e30], bias_zeros) 
        # self.biases = torch.tensor(dummy)
    
        # self.sense_weight_logits = np.zeros_like(sense_mask)
        
        
        
        #mean_prob = sense_mask / np.sum(sense_mask, axis=1, keepdims=True)

        #mean_qp = torch.tensor(mean_prob)
        #self.mean_qp = torch.reshape(
        #    mean_qp,
        #    (self.vocab.size, self.max_senses), 
        #)

        #mean_qd = torch.tensor(mean_prob)
        #self.mean_qd = torch.reshape(
        #    mean_qd, 
        #    (self.vocab.size, self.max_senses), 
        #)
        
        #self.update_mean_qp = self.update_smoothed_mean(
        #        self.mean_qp, qp,
        #        indices=torch.unsqueeze(self.targets, 1)
        #)

        #self.loss = self.lm_loss + self.d_loss + self.m_loss
        #self.add_find_neighbours()
        #self.add_get_mean_q()

        #self.learning_rate = options.learning_rate
        #self.opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    #def forward(self, input_ids, attention_mask=None, token_type_ids=None):
    #    outputs = self.disambiguation_bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    #    sequence_output = outputs.last_hidden_state
    #    logits = torch.matmul(sequence_output, self.embeddings.t()) + self.biases_with_dummy
    #    return logits

    def train_model(self, corpus):
        for batch_num, batch_data in enumerate(corpus):
            self.optimizer.zero_grad()
            loss = self.forward(batch_data)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            print(f'Batch {batch_num}, Loss {loss.item()}')
    
    def setup_sense_indices(self, n_senses, total_senses):
        sense_indices = np.zeros((self.vocab.size, self.max_senses), dtype=np.int32)
        index = 1
        for i, n in enumerate(n_senses):
            for j in range(n):
                sense_indices[i, j] = index
                index += 1
        return sense_indices
        
    #def get_sense_embeddings(self, tokens):
    #    sense_indices = torch.nn.functional.embedding(tokens, self.sense_indices)
    #    return torch.nn.functional.embedding(sense_indices, self.embeddings)

    def get_sense_embeddings_and_biases(self, tokens):
        sense_indices = torch.nn.functional.embedding(tokens, self.sense_indices)
        sense_embeddings = torch.nn.functional.embedding(sense_indices, self.embeddings)
        sense_biases = torch.nn.functional.embedding(sense_indices, self.biases_with_dummy)
        return sense_embeddings, sense_biases

    def make_word_embeddings(self, seqs, sense_weights=None):
        # ids.shape + (n_senses, embedding_size)
        sense_embeddings = self.sense_embeddings(seqs)
        if sense_weights is None:
            sense_weight_logits = torch.nn.functional.embedding(seqs, self.sense_weight_logits)
            sense_mask = torch.nn.functional.embedding(seqs, self.sense_mask)
            sense_weights = masked_softmax(sense_weight_logits, sense_mask)

        weighted_sense_embeddings = torch.sum(sense_embeddings * sense_weights.unsqueeze(-1), dim=-2)
        return weighted_sense_embeddings

    def calculate_sense_probs(self, seqs, reps):
        sense_embeddings, sense_biases = self.get_sense_embeddings_and_biases(seqs)
        sense_scores = torch.matmul(sense_embeddings, torch.unsqueeze(reps, dim=-1))
        sense_scores = torch.squeeze(sense_scores, dim = -1)
        sense_scores += sense_biases
        sense_mask = torch.nn.functional.embedding(seqs, self.sense_mask)
        sense_probs = masked_softmax(sense_scores, sense_mask) 
        return sense_probs
    
    def disambiguation_layer(self, seqs):
        word_embeddings = self.make_word_embeddings(seqs)
        reps = self.disambiguation_bert(word_embeddings, self.padding)
        sense_probs = self.calculate_sense_probs(seqs, reps)
        disambiguated_reps = self.make_word_embeddings(seqs, sense_weights=sense_probs)
        return disambiguated_reps, sense_probs

    def prediction_layer(self, reps):
        return self.prediction_bert(reps, self.padding)
            
    #def update_smoothed_mean(self, mean, values, indices=None, weight=0.005):
    #    if indices is None:
    #        mean.data = (1.0 - weight) * mean + weight * values
    #        return mean
        
        # TODO: if this breaks, try
        # current_values = mean[indices]
        # updates = weight * (values - current_values)
        # mean[indices]
        #current_values = torch.gather(mean, indices)
        #updates = weight * (values - current_values)
        #return mean.scatter_add_(0, indices, updates)

    #def add_to_feed_dict(self, feed_dict, batch, dl_r, ml_coeff):
    #    padding = np.zeros(batch.unmasked_seqs.shape, dtype=np.int32)
    #    for i, l in enumerate(batch.seq_len):
    #        padding[i, :l] = 1
    #    feed_dict.update({
    #            self.unmasked_seqs: batch.unmasked_seqs,
    #            self.masked_seqs: batch.masked_seqs,
    #            self.padding: padding,
    #            self.target_positions: batch.target_positions,
    #            self.targets: batch.targets,
    #            self.dl_r: dl_r,
    #            self.ml_coeff: ml_coeff})

    #def contextualize(self, batch):
    #    padding = np.zeros(batch.masked_seqs.shape, dtype=np.int32)
    #    for i, l in enumerate(batch.seq_len):
    #        padding[i, :l] = 1
        
    #    # TODO: this was previously in a dict that was passed into the session
    #    # might have conflicts
    #    self.masked_seqs = batch.masked_seqs,
    #    self.padding = padding,
    #    self.target_position = batch.target_positions
        
    #    return self.target_reps

    #def disambiguate(self, batch, method='prediction'):
    #    padding = np.zeros(batch.masked_seqs.shape, dtype=np.int32)
    #    for i, l in enumerate(batch.seq_len):
    #        padding[i, :l] = 1
            
    #    self.unmasked_seqs = batch.unmasked_seqs
    #    self.masked_seqs = batch.masked_seqs
    #    self.padding = padding
    #    self.target_positions = batch.target_positions
    #    self.targets = batch.targets
        
    #    prob_tensors = {'prediction': self.qp}
    #    if self.has_disambiguation_layer:
    #        prob_tensors['disambiguation'] = self.qd
            
    #    return prob_tesnsors[method]

    #def get_target_probs(self, batch):
    #    padding = np.zeros(batch.masked_seqs.shape, dtype=np.int32)
    #    for i, l in enumerate(batch.seq_len):
    #        padding[i, :l] = 1
        
    #    self.unmasked_seqs = batch.unmasked_seqs
    #    self.masked_seqs = batch.masked_seqs
    #    self.padding = padding
    #    self.target_positions = batch.target_positions
    #    self.targets = batch.targets
                
    #    return [self.target_token_probs, self.target_sense_probs]

    #def add_get_mean_q(self):
    #    self.mean_q_tokens =  torch.empty((0), dtype=torch.int32)
    #    self.selected_mean_qp = torch.nn.functional.embedding(self.mean_q_tokens, self.mean_qp)
    #    self.selected_mean_qd = torch.nn.functional.embedding(self.mean_q_tokens, self.mean_qd)

    #def get_mean_sense_probs(self, tokens):
    #    request = {
    #        'qp': self.selected_mean_qp,
    #        'qd': self.selected_mean_qd,
    #    }
    #    self.mean_q_tokens = tokens
    #    return request

    # On the TODO
    def add_find_neighbours(self):
        self.interesting_ids = torch.empty((0), dtype=torch.int32)
        self.n_neighbours = torch.tensor([], dtype=torch.int32)

        sense_indices = torch.nn.functional.embedding(self.interesting_ids, self.sense_indices) 
        interesting_embeddings =  torch.nn.functional.embedding(sense_indices, self.embeddings)
        interesting_embeddings = torch.reshape(interesting_embeddings, [-1, self.embedding_size])
        interesting_norms = torch.norm(interesting_embeddings, dim=1)

        norms = torch.norm(self.embeddings, dim=1)

        # (n_interesting, vocab.size*n_senses)
        dot = torch.matmul(interesting_embeddings,
                        torch.transpose(self.embeddings,0,1))
        dot = dot / torch.unsqueeze(interesting_norms, dim=1)
        dot = dot / torch.unsqueeze(norms, dim=0)
        cosine_similarities = torch.reshape(
                dot,
                [-1, self.max_senses, self.total_senses])
        mask = torch.concat([
                torch.tensor([2.0]),
                torch.zeros([self.total_senses - 1], dtype=torch.float32)],
                dim=0)
        mask = torch.unsqueeze(mask, dim=0)
        mask = torch.unsqueeze(mask, dim=0)
        cosine_similarities = cosine_similarities - mask

        self.neighbour_similarities, indices = torch.top_k(
                cosine_similarities, k=self.n_neighbours)
        self.neighbour_similarities = self.neighbour_similarities[:, :, 1:]
        self.neighbour_tokens = torch.nn.functional.embedding(indices,
                self.sense_to_token)[:, :, 1:]
        self.neighbour_sense_nums = torch.nn.functional.embedding(indices,
                self.sense_to_sense_num)[:, :, 1:]

    def get_neighbours(self, tokens, n=10):
        self.interesting_ids = tokens
        self.n_neighbours = n
        return self.neighbour_similarities, self.neighbour_tokens, self.neighbour_sense_nums

def deduplicated_indexed_slices(values, indices):
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
    def __init__(self, vocab, options, multisense_vocab={}, training=False):
        super(PolyLM, self).__init__()
        
        self.vocab = vocab
        self.options = options
        self.training = training
        self.max_seq_len = self.options.max_seq_len
        self.embedding_size = self.options.embedding_size
        self.max_senses = self.options.max_senses_per_word

        gpus = [int(x) for x in self.options.gpus.split(',')]
        logging.info('Building PolyLM on GPU(s) ' + ', '.join([str(x) for x in gpus]))
        self.n_towers = len(gpus)

        self.global_step = torch.ones(1, dtype=torch.int32)

        self.n_senses = np.ones([self.vocab.size], dtype=np.int32)
        for t, n in multisense_vocab.items():
            assert n > 0 and n <= self.max_senses
            self.n_senses[t] = n

        self.towers = []
        self.grads = []
        self.losses = []
        self.lm_losses = []
        self.d_losses = []
        self.m_losses = []

        for i in range(self.n_towers):
            #with torch.device('/gpu:%d' % i):
                #will need to change this as it has variable scope into another class system. 
                #with tf.variable_scope('polylm', reuse=tf.AUTO_REUSE):
                tower = PolyLMModel(
                        self.vocab, self.n_senses,
                        self.options, training=training)
                
                self.towers.append(tower)
                self.losses.append(0)
                self.grads.append(0)
                self.lm_losses.append(0)
                self.d_losses.append(0)
                self.m_losses.append(0)

        self.default_model = self.towers[0]
        self.loss = torch.mean(torch.stack(self.losses))
        self.lm_loss = torch.mean(torch.stack(self.lm_losses))
        self.d_loss = torch.mean(torch.stack(self.d_losses))
        self.m_loss = torch.mean(torch.stack(self.m_losses))
        
        grads_and_vars = average_gradients(self.grads)
        clipped_grads, self.grad_norm = clip_gradients(grads_and_vars, self.options.max_gradient_norm)
        self.update_params = tower.opt.apply_gradients(clipped_grads, global_step=self.global_step)
       
        self.update_params
        self.default_model.update_mean_qp
        self.default_model.update_mean_qd 
        
        # self.saver = torch.save(self.default_model.state_dict(), 'checkpoint.pth')

    # NOTE: removed attempt_restore, get_embeddings, get_masked

    def train_on_batch(self, batches, step_num):
        assert len(batches) == self.n_towers
        if step_num < self.options.lr_warmup_steps:
            lr_ratio = (step_num + 1) / self.options.lr_warmup_steps
        elif self.options.anneal_lr:
            lr_ratio = (self.options.n_batches -
                        step_num) / self.options.n_batches
        else:
            lr_ratio = 1.0
        learning_rate = lr_ratio * self.options.learning_rate

        if step_num < self.options.dl_warmup_steps:
            dl_ratio = step_num / self.options.dl_warmup_steps
            dl_r = 1.0 + dl_ratio * (self.options.dl_r - 1.0)
        else:
            dl_r = self.options.dl_r

        if step_num < self.options.ml_warmup_steps:
            ml_ratio = step_num / self.options.ml_warmup_steps
            ml_coeff = ml_ratio * self.options.ml_coeff
        else:
            ml_coeff = self.options.ml_coeff

        #feed_dict = {self.learning_rate: learning_rate}
        #for i, batch in enumerate(batches):
        #    self.towers[i].add_to_feed_dict(feed_dict, batch, dl_r, ml_coeff)
        
        
        #def forward(unmasked_seqs, masked_seqs, padding, target_positions, targets, dl_r, ml_coeff):

        start_time = time.time()
        self.optimizer.zero_grad()
        for i, batch in enumerate(batches):
            loss, lm_loss, d_loss, m_loss = self._towers[i].forward(batches.unmasked_seqs, batches.masked_seqs, _, batches.target_positions, batches.targets, dl_r, ml_coeff)

         
        loss = self.loss 
        loss.backward()
        self.optimizer.step()
        end_time = time.time()
        
        batch_output = {
            "time": end_time - start_time,
            "lr": learning_rate,
            "dl_r": dl_r,
            "ml_coeff": ml_coeff,
            "loss": loss.item(),
            "lm_loss": self.lm_loss.item(),
            "d_loss": self.d_loss.item(),
            "m_loss": self.m_loss.item(),
            "global_step": self.global_step,
            "grad_norm": torch.norm(self.model.parameters()).item(),
        }

        return batch_output
    
    # TODO: remove get_n_senses, get_sense_probs

    #def get_n_senses(self, vocab_id):
    #    return self.n_senses[vocab_id]

    #def get_sense_probs(self, vocab_ids):
    #    return self.default_model.get_mean_sense_probs(vocab_ids)[0]
    

    def train(self, corpus, test_words=[]):
        logging.info('Commencing training...')
        logging.info('Vocab size: %d' % self.vocab.size)
        logging.info('Total senses: %d' % np.sum(self.n_senses))

        checkpoint_path = os.path.join(self.options.model_dir, 'polylm.ckpt')
        test_tokens = [self.vocab.str2id(w) for w in test_words]
        
        #self.display_words(test_words)
        
        gpu_time_for_block = 0.0
        loss_for_block = 0.0
        lm_loss_for_block = 0.0
        d_loss_for_block = 0.0
        m_loss_for_block = 0.0
        norm_for_block = 0.0
        #global_step = 1
        tokens_read = 0
        masking_policy = [
                float(x) for x in self.options.masking_policy.split()]
        logging.info('Masking policy: %.2f [MASK], %.2f self, %.2f random' % (
                masking_policy[0],
                masking_policy[1],
                masking_policy[2]))
        batch_gen = corpus.generate_batches(
                self.options.batch_size, self.options.max_seq_len,
                self.options.mask_prob, variable_length=True,
                masking_policy=masking_policy)
        batch_num = self.global_step
        block_start_batch = batch_num
        block_start_time = time.time()
        while batch_num < self.options.n_batches:
            batch_num += 1
            batches = []
            for i in range(self.n_towers):
                batch = next(batch_gen)
                batches.append(batch)
                tokens_read += batch.n_tokens()

            batch_output = self.train_on_batch(batches, batch_num)

            gpu_time_for_block += batch_output['time']
            loss_for_block += batch_output['loss']
            lm_loss_for_block += batch_output['lm_loss']
            d_loss_for_block += batch_output['d_loss']
            m_loss_for_block += batch_output['m_loss']
            norm_for_block += batch_output['grad_norm']

            if batch_num % self.options.print_every == 0:
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

            if batch_num % self.options.test_every == 0:
                #self.display_words(test_words)
                logging.info(
                        'lr = %.8f, dl_r = %.5f, ml_coeff = %.5f' % (
                                batch_output['lr'],
                                batch_output['dl_r'],
                                batch_output['ml_coeff']))

            # if batch_num % self.options.save_every == 0:
                # logging.info('Saving to %s...' % checkpoint_path)
                # self.saver.save(checkpoint_path, global_step=self.global_step)


    #def disambiguate(self, batch, method='prediction'):
    #    return self.default_model.disambiguate(batch, method=method)

    #def contextualize(self, batch):
    #    return self.default_model.contextualize(batch)

    #def get_target_probs(self, batch):
    #    return self.default_model.get_target_probs(batch)

    #def match(self, keys, values):
    #    return self.default_model.match(keys, values)

    #def get_top_k_substitutes(self, reps, k):
    #    return self.default_model.get_top_k_substitutes(reps, k)

    #def display_words(self, words):
    #    vocab_ids = [self.vocab.str2id(t) for t in words]
    #    words = [self.vocab.id2str(i) for i in vocab_ids]
    #    sense_stats = self.default_model.get_mean_sense_probs(vocab_ids)
    #    similarities, tokens, senses = self.default_model.get_neighbours(vocab_ids)
        
    #    for i, (vocab_id, word) in enumerate(zip(vocab_ids, words)):
    #        info = [
    #                'qd=%.4f, qp=%.4f' % (
    #                        sense_stats['qd'][i, s],
    #                        sense_stats['qp'][i, s])
    #                for s in range(self.n_senses[vocab_id])]
    #        util.display_word(self.vocab, word, similarities[i, :, :],
    #                          tokens[i, :, :], senses[i, :, :],
    #                          self.n_senses[vocab_id], info=info)