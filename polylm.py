#import init
import math
import logging
import os
import sys
import time

import torch.nn
import torch
import numpy as np

from tfbert import BertModel, BertConfig
import util

is_tf = True
if is_tf:
    import tensorflow as tf

# model = PolyLMModel(vocab, n_senses, options, training=True)
# logits = model(input_ids, attention_mask, token_type_ids)
# loss = model.calculate_loss(logits, targets)

def masked_softmax(logits, mask):
    masked_logits = logits - 1e30 * (1.0 - mask)
    return torch.nn.functional.softmax(masked_logits, dim=-1)

class PolyLMModel(torch.nn.Module):
    def __init__(self, vocab, n_senses, options, training=False):
        super(PolyLMModel, self).__init__()

        # Initialization
        self.vocab = vocab
        self.options = options
        self.embedding_size = options.embedding_size
        self.max_seq_len = options.max_seq_len
        self.max_senses = options.max_senses_per_word
        self.training = training
        self.has_disambiguation_layer = (options.use_disambiguation_layer and self.max_senses > 1)
        
        # Set up BERT
        self.disambiguation_bert_config = BertConfig(
            hidden_size=self.embedding_size,
            num_hidden_layers=options.n_disambiguation_layers,
            intermediate_size=options.bert_intermediate_size,
            num_attention_heads=options.n_attention_heads,
            hidden_dropout_prob=options.dropout,
            attention_probs_dropout_prob=options.dropout,
            max_position_embeddings=self.max_seq_len,
        )

        self.prediction_bert_config = BertConfig(
            hidden_size=self.embedding_size,
            num_hidden_layers=options.n_prediction_layers,
            intermediate_size=options.bert_intermediate_size,
            num_attention_heads=options.n_attention_heads,
            hidden_dropout_prob=options.dropout,
            attention_probs_dropout_prob=options.dropout,
            max_position_embeddings=self.max_seq_len,
        )
        
        # NOTE: unmasked_seqs, masked_seqs, padding, targets, target_positions, dl_r, and ml_coeff are passed directly into the forward function by corpus.generate_batches()
        # We do not need to initialize placeholders for them here
        
        # SENSE INDICES AND MASKS
        self.total_senses = n_senses.sum() + 1
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
        self.n_senses = torch.tensor(n_senses)
        #self.sense_to_token = torch.tensor(sense_to_token)
        #self.sense_to_sense_num = torch.tensor(sense_to_sense_num)

        # EMBEDDINGS AND BIASES
        # Used the embedding class instead of initializing parameters
        self.embeddings = torch.nn.Embedding(
            num_embeddings=self.total_senses, 
            embedding_dim=self.embedding_size
        ) 
        self.embeddings.weight.data = torch.nn.init.normal_(torch.empty(self.total_senses, self.embedding_size))
        
        self.biases = torch.nn.Parameter(torch.zeros(self.total_senses - 1)) 
        self.biases_with_dummy = torch.cat([torch.tensor([-1e30]), self.biases]) 
        self.sense_weight_logits = torch.nn.Parameter(torch.zeros(self.vocab.size, self.max_senses))
                
        no_predict_tokens = [
            self.vocab.bos_vocab_id,
            self.vocab.eos_vocab_id,
            self.vocab.pad_vocab_id,
            self.vocab.mask_vocab_id,
        ]
        
        unpredictable_tokens = np.zeros([self.total_senses], dtype=np.float32)
        for t in no_predict_tokens:
            unpredictable_tokens[self.sense_indices[t, 0]] = 1.0
        self.unpredictable_tokens = torch.tensor(unpredictable_tokens)
        
        # NOTE: Everything else from init is placed in the forward function
            
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
        
        self.learning_rate = options.learning_rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, unmasked_seqs, masked_seqs, padding, target_positions, targets, dl_r, ml_coeff):
        
        # TODO: Need this for some reason
        self.embeddings = torch.nn.Embedding(
            num_embeddings=self.total_senses, 
            embedding_dim=self.embedding_size
        ) 
        self.embeddings.weight.data = torch.nn.init.normal_(torch.empty(self.total_senses, self.embedding_size))
        
        # LANGUAGE MODEL LOSS
        self.padding = padding # This is used in BERT 
        
        if self.options.use_disambiguation_layer:
            disambiguated_reps, _ = self.disambiguation_layer(masked_seqs)
            _, qd = self.disambiguation_layer(unmasked_seqs)
            qd = qd.view(-1, self.max_senses)
            qd = qd[target_positions]
        else:
            pass
       
        output_reps = self.prediction_layer(disambiguated_reps)
        flattened_reps = output_reps.view(-1, self.embedding_size) 
        target_reps = flattened_reps[target_positions] # (n_targets, embedding_size)

        target_position_scores = torch.matmul(target_reps, self.embeddings.weight.transpose(0, 1)) + self.biases_with_dummy.unsqueeze(0) - 1e30 * self.unpredictable_tokens.unsqueeze(0) # (n_targets, total_senses)
                
        target_position_probs = torch.nn.functional.softmax(target_position_scores, dim=1)
        target_sense_indices = self.sense_indices[targets]
        target_sense_indices = target_sense_indices.type(torch.int64)
        target_sense_probs = torch.gather(target_position_probs, 1, target_sense_indices)

        target_sense_masks = self.sense_mask[targets]
        target_sense_probs *= target_sense_masks
        
        target_token_probs = torch.sum(target_sense_probs, dim=1)
        target_token_probs = torch.clamp(target_token_probs, min=1e-30)
        log_target_probs = torch.log(target_token_probs + 1e-30)
        
        lm_loss = -torch.mean(log_target_probs)

        # DISAMBIGUATION LOSS
        qp = target_sense_probs / target_token_probs.unsqueeze(1)
        qp = torch.clamp(qp, min=0)
        targets_are_multisense = self.is_multisense[targets]
        n_multisense = torch.sum(targets_are_multisense) + 1e-6
        
        sharpened_q = torch.pow(qp, dl_r)
        eps=1e-7
        sharpened_q = torch.nn.functional.relu(sharpened_q)
        log_sharpened_q = torch.log(torch.sum(sharpened_q + eps, dim=1))
        log_sharpened_q *= targets_are_multisense

        d_loss = -torch.sum(log_sharpened_q) / (dl_r * n_multisense)

        # METRIC LOSS
        if self.has_disambiguation_layer:
            # NOTE: ignored qp.stop_gradient, update_mean_qd
            p_norms = torch.norm(qp, dim=1)
            d_norms = torch.norm(qd, dim=1)
            cosine_sim = torch.sum(qp * qd, dim=1) / (p_norms * d_norms + 1e-10)
            cosine_sim *= targets_are_multisense
            m_loss = -ml_coeff * torch.sum(cosine_sim) / n_multisense
        else:
            m_loss = torch.tensor(0.0)
            
         # NOTE: ignored update_mean_qd, add_find_neighbors, add_get_mean_q
        
        total_loss = lm_loss + d_loss + m_loss
        return total_loss, lm_loss, d_loss, m_loss
            
    def get_sense_embeddings(self, tokens):
        sense_indices = self.sense_indices[tokens]
        sense_embeddings = self.embeddings(sense_indices)
        return sense_embeddings

    def get_sense_embeddings_and_biases(self, tokens):
        sense_indices = self.sense_indices[tokens]
        sense_embeddings = self.embeddings(sense_indices)
        sense_biases = self.biases_with_dummy[sense_indices] 
        return sense_embeddings, sense_biases

    def make_word_embeddings(self, seqs, sense_weights=None):
        # ids.shape + (n_senses, embedding_size)
        sense_embeddings = self.get_sense_embeddings(seqs)

        if sense_weights is None:
            # ids.shape + (n_senses,)
            sense_weight_logits = self.sense_weight_logits[seqs]
            sense_mask = self.sense_mask[seqs]
            sense_weights = masked_softmax(sense_weight_logits, sense_mask)

        # TODO: Check weighted_sense_embeddings
        weighted_sense_embeddings = torch.sum(sense_embeddings * sense_weights.unsqueeze(-1), dim=-2)
        return weighted_sense_embeddings
    
    def calculate_sense_probs(self, seqs, reps):
        # ids.shape + (n_senses, embedding_size)
        sense_embeddings, sense_biases = self.get_sense_embeddings_and_biases(seqs)
        # ids.shape + (n_senses, 1)
        sense_scores = torch.matmul(sense_embeddings, torch.unsqueeze(reps, dim=-1))
        # ids.shape + (n_senses)
        sense_scores = torch.squeeze(sense_scores, dim = -1)
        sense_scores += sense_biases
        sense_mask = torch.nn.functional.embedding(seqs, self.sense_mask)
        sense_probs = masked_softmax(sense_scores, sense_mask) 
        return sense_probs
    
    def disambiguation_layer(self, seqs):
        word_embeddings = self.make_word_embeddings(seqs)

        if is_tf:
            word_embeddings = tf.convert_to_tensor(word_embeddings.detach().numpy())
            padding = tf.convert_to_tensor(self.padding)
            #padding = tf.convert_to_tensor(self.padding.detach().numpy())
        
            model = BertModel(
                config=self.disambiguation_bert_config,
                is_training=self.training,
                input_embeddings=word_embeddings,
                input_mask=padding,
            )
            
            reps = model.get_output() # (batch_size, sentence_len, embedding_size)
            reps = torch.from_numpy(reps.numpy())
        
        else:
            #attention_mask = (1 - self.padding).unsqueeze(1).unsqueeze(2)
            #attention_mask = attention_mask.to(dtype=torch.float32)
            #attention_mask = (1.0 - attention_mask) * -10000.0 

            disambiguation_bert = BertModel(self.disambiguation_bert_config)
            reps = disambiguation_bert(word_embeddings, padding)

        # (batch_size, sentence_len, n_senses)
        sense_probs = self.calculate_sense_probs(seqs, reps) 
        # (batch_size, sentence_len, embedding_size)
        disambiguated_reps = self.make_word_embeddings(seqs, sense_weights=sense_probs)
    
        return disambiguated_reps, sense_probs

    def prediction_layer(self, reps):
        if is_tf:
            reps = tf.convert_to_tensor(reps.detach().numpy())
            padding = tf.convert_to_tensor(self.padding)
            #padding = tf.convert_to_tensor(self.padding.detach().numpy())
            
            model = BertModel(
                config=self.prediction_bert_config,
                is_training=self.training,
                input_embeddings=reps,
                input_mask=padding,
            )
            reps = model.get_output()
            reps = torch.from_numpy(reps.numpy())
        else: 
            prediction_bert = BertModel(self.prediction_bert_config)
            reps = prediction_bert(reps, padding)

        return reps
            
    # NOTE: update_smoothed_mean is used in _update_mean_qd and _update_mean_qp
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

    # NOTE: contextualize, disambiguate, get_target_probs add_get_mean_q, get_mean_sense_probs, get_sense_embeddings not called
    
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

    # NOTE: this is called in the forward funciton later
    
    #def add_find_neighbours(self):
    #    self.interesting_ids = torch.empty((0), dtype=torch.int32)
    #    self.n_neighbours = torch.tensor([], dtype=torch.int32)

    #    sense_indices = torch.nn.functional.embedding(self.interesting_ids, self.sense_indices) 
    #    interesting_embeddings =  torch.nn.functional.embedding(sense_indices, self.embeddings)
    #    interesting_embeddings = torch.reshape(interesting_embeddings, [-1, self.embedding_size])
    #    interesting_norms = torch.norm(interesting_embeddings, dim=1)

    #    norms = torch.norm(self.embeddings, dim=1)

    #    # (n_interesting, vocab.size*n_senses)
    #    dot = torch.matmul(interesting_embeddings,
    #                    torch.transpose(self.embeddings,0,1))
    #    dot = dot / torch.unsqueeze(interesting_norms, dim=1)
    #    dot = dot / torch.unsqueeze(norms, dim=0)
    #    cosine_similarities = torch.reshape(
    #            dot,
    #            [-1, self.max_senses, self.total_senses])
    #    mask = torch.concat([
    #            torch.tensor([2.0]),
    #            torch.zeros([self.total_senses - 1], dtype=torch.float32)],
    #            dim=0)
    #    mask = torch.unsqueeze(mask, dim=0)
    #    mask = torch.unsqueeze(mask, dim=0)
    #    cosine_similarities = cosine_similarities - mask

    #    self.neighbour_similarities, indices = torch.top_k(
    #            cosine_similarities, k=self.n_neighbours)
    #    self.neighbour_similarities = self.neighbour_similarities[:, :, 1:]
    #    self.neighbour_tokens = torch.nn.functional.embedding(indices,
    #            self.sense_to_token)[:, :, 1:]
    #    self.neighbour_sense_nums = torch.nn.functional.embedding(indices,
    #            self.sense_to_sense_num)[:, :, 1:]

    #def get_neighbours(self, tokens, n=10):
    #    self.interesting_ids = tokens
    #    self.n_neighbours = n
    #    return self.neighbour_similarities, self.neighbour_tokens, self.neighbour_sense_nums

# NOTE: deduplicated_indexed_slices, average_gradients, and clip_gradients are only needed for manual gradient descent

#def deduplicated_indexed_slices(values, indices):
#    # Calculate unique indices and their counts
#    unique_indices, inverse_indices = torch.unique(indices, return_inverse=True)
#    counts = torch.bincount(inverse_indices)

#    # Calculate the summed values using scatter_add
#    summed_values = torch.zeros_like(unique_indices, dtype=values.dtype)
#    summed_values.scatter_add_(0, inverse_indices.unsqueeze(0).expand_as(values), values)

#    return summed_values, unique_indices

#def average_gradients(tower_grads):
#    average_grads = []
#    for grad_and_vars in zip(*tower_grads):
#        g0, v0 = grad_and_vars[0]

#        if g0 is None:
#            average_grads.append((g0, v0))
#            continue

#        if isinstance(g0, torch.sparse_coo_tensor):
#            indices = []
#            values = []
#            for g, v in grad_and_vars:
#                indices.append(g.indices)
#                values.append(g.values)
#            all_indices = torch.cat(indices, dim=0)
#            avg_values = torch.cat(values, dim=0) / len(grad_and_vars)
#            av, ai = _deduplicated_indexed_slices(avg_values, all_indices)
#            grad = torch.sparse_coo_tensor(av, ai, g0.size())
#        else:
#            grads = []
#            for g, _ in grad_and_vars:
#                expanded_g = torch.unsqueeze(g, 0)
#                grads.append(expanded_g)

#            grad = torch.cat(grads, dim=0)
#            grad = torch.mean(grad, dim=0)

#        average_grads.append((grad, v0))

#    return average_grads

#def clip_gradients(grads_and_vars, val):
#    grads = [g for g, v in grads_and_vars]
#    var = [v for g, v in grads_and_vars]
#    clipped_grads, grad_norm = torch.nn.utils.clip_grad_norm_ (grads, val)
#    return list(zip(clipped_grads, var)), grad_norm

class PolyLM(torch.nn.Module):
    def __init__(self, vocab, options, multisense_vocab={}, training=False):
        super(PolyLM, self).__init__()
        self.vocab = vocab
        self.options = options
        self.max_seq_len = options.max_seq_len
        self.embedding_size = options.embedding_size
        self.max_senses = options.max_senses_per_word
        self.training = training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize sense numbers
        self.n_senses = torch.ones(vocab.size, dtype=torch.int32)
        for t, n in multisense_vocab.items():
            assert 0 < n <= self.max_senses
            self.n_senses[t] = n

        ## Model setup across CPUs
        self.models = torch.nn.ModuleList([
            PolyLMModel(vocab, self.n_senses, options, training=training).to('cpu')
            for _ in range(len(options.gpus.split(",")))  # Number of models to instantiate
        ])

        ## Model setup across GPUs
        #self.models = torch.nn.ModuleList([
        #    PolyLMModel(vocab, self.n_senses, options, training=training).to(f'cuda:{i}')
        #    for i in range(len(options.gpus.split(",")))
        #])
        
        self.global_step = 0 # global_step = torch.ones(1, dtype=torch.int32)
        #self.optimizer = torch.nn.optim.Adam(self.parameters(), lr=options.learning_rate)
    
    #def train_model(self, corpus):
    #    for batch_num, batch_data in enumerate(corpus):
    #        self.optimizer.zero_grad()
    #        loss = self.forward(batch_data)
    #        loss.backward()
    #        self.optimizer.step()
    #        self.scheduler.step()
    #        print(f'Batch {batch_num}, Loss {loss.item()}')

  
    def train_model(self, corpus, num_epochs):
        # Initialize model and optimizer
        model = PolyLMModel(self.vocab, self.n_senses, self.options, training=self.training)
        model.to(self.device)
        options = model.options
        optimizer = model.optimizer
        
        masking_policy = [float(x) for x in options.masking_policy.split()]
        batches = corpus.generate_batches(
            options.batch_size, 
            options.max_seq_len, 
            options.mask_prob,
            variable_length=True,
            masking_policy=masking_policy)
         
        # Training loop

        for epoch in range(num_epochs):
            total_loss = 0
            batch_count = 0
            for batch in batches:
                unmasked_seqs = torch.tensor(batch.unmasked_seqs, dtype=torch.long, device=self.device)
                masked_seqs = torch.tensor(batch.masked_seqs, dtype=torch.long, device=self.device)
                target_positions = torch.tensor(batch.target_positions, dtype=torch.long, device=self.device)
                targets = torch.tensor(batch.targets, dtype=torch.long, device=self.device)
                
                padding = np.zeros(unmasked_seqs.shape, dtype=np.int32)
                for i, l in enumerate(batch.seq_len):
                    padding[i, :l] = 1
            
                # Forward pass
                optimizer.zero_grad()
                dl_r = 0.5  # Example disambiguation layer rate
                ml_coeff = 0.1  # Example metric loss coefficient
                
                total_loss, lm_loss, d_loss, m_loss = model.forward(unmasked_seqs, masked_seqs, padding, target_positions, targets, dl_r, ml_coeff)
                
                # Backward and optimize
                total_loss.backward()
                optimizer.step()

                # Loss accumulation
                total_loss += total_loss.item()
                batch_count += 1
            
            # Logging
                if batch_count > 0:
                    average_loss = total_loss / batch_count
                    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {average_loss:.4f}')
                else:
                    print("No batches processed.")

            #saving:
                if (batch_count % options.save_every == 0):
                    print(model.state_dict())
                    torch.save(model.state_dict(), 'models/save_number' + str(batch_count))

        print('Finished Training')

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
