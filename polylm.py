import torch.nn
import torch
import numpy as np

is_tf = False
debug = False

if is_tf:
    import tensorflow as tf
    from tfbert import BertModel, BertConfig
else:
    from bert import BertModel, BertConfig

def check(*args):
    for tensor in args:
        not_nan(tensor)
    
def not_nan(tensor):
    assert not torch.isnan(tensor).any(), "Tensor contains NaNs!"

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
        self.options = options
        
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
        # NOTE: Removed sense_to_token, sense_to_sense_num
        
        # SENSE INDICES AND MASKS
        self.total_senses = n_senses.sum() + 1
        sense_indices = np.zeros([self.vocab.size, self.max_senses], dtype=np.int32)
        sense_mask = np.zeros([self.vocab.size, self.max_senses], dtype=np.float32)
        is_multisense = np.zeros([self.vocab.size], dtype=np.float32)

        index = 1
        for i, num_senses in enumerate(n_senses):
            if num_senses > 1:
                is_multisense[i] = 1.0
            for j in range(num_senses):
                sense_indices[i, j] = index
                sense_mask[i, j] = 1.0
                index += 1

        self.register_buffer('sense_indices', torch.from_numpy(sense_indices))
        self.register_buffer('sense_mask', torch.from_numpy(sense_mask))
        self.register_buffer('is_multisense', torch.from_numpy(is_multisense))
        self.n_senses = torch.tensor(n_senses)

        # EMBEDDINGS AND BIASES
        # Used the embedding class instead of initializing parameters
        self.embeddings = torch.nn.Embedding(
            num_embeddings=self.total_senses, 
            embedding_dim=self.embedding_size
        )
        
        assert self.embedding_size > 0
        self.embeddings.weight.data = torch.nn.init.normal_(
            torch.empty(self.total_senses, self.embedding_size), 
            0.0,
            1.0 / np.sqrt(self.embedding_size)
        )
        
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
        # Removed mean_qp, mean_qd, update_mean_qp
        
        self.learning_rate = options.learning_rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, unmasked_seqs, masked_seqs, padding, target_positions, targets, dl_r, ml_coeff):
        # Safety assertions
        check(unmasked_seqs, masked_seqs, target_positions, targets)
        check(self.biases, self.sense_weight_logits, self.embeddings.weight)
        
        self.padding = padding
        check(torch.tensor(self.padding))
                
        # LANGUAGE MODEL LOSS
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
            qp.detach()
            p_norms = torch.norm(qp, dim=1)
            d_norms = torch.norm(qd, dim=1)
            cosine_sum = torch.sum(qp * qd, dim=1) / (p_norms * d_norms + 1e-10)
            cosine_sum *= targets_are_multisense
            m_loss = -ml_coeff * torch.sum(cosine_sum) / n_multisense
        else:
            m_loss = torch.tensor(0.0)

        # NOTE: ignored update_mean_qd, add_find_neighbors, add_get_mean_q
         
        total_loss = lm_loss + d_loss + m_loss
        assert not torch.isnan(total_loss).any(), "Tensor contains NaNs!"
        return total_loss
    
    def run_forward(self, unmasked_seqs, masked_seqs, padding, target_positions, targets, dl_r, ml_coeff):
        # Safety assertions
        check(unmasked_seqs, masked_seqs, target_positions, targets)
        check(self.biases, self.sense_weight_logits, self.embeddings.weight)
        
        self.padding = padding
        check(torch.tensor(self.padding))
                
        # LANGUAGE MODEL LOSS
        if self.options.use_disambiguation_layer:
            disambiguated_reps, _ = self.disambiguation_layer(masked_seqs)
            _, qd = self.disambiguation_layer(unmasked_seqs)
            qd = qd.view(-1, self.max_senses)
            qd = qd[target_positions]
        else:
            pass
       
        output_reps = self.prediction_layer(disambiguated_reps)
        return output_reps
            
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
        # seqs is torch.Size([64, 16]), which comes from batch.masked_seqs
        # sense_weight_logits is torch.Size([1138, 8]), which is batch_size by max_seq_len * self._vocab.pad_vocab_id
        
        # ids.shape + (n_senses, embedding_size)
        sense_embeddings = self.get_sense_embeddings(seqs)
        check(seqs, sense_embeddings)        

        if sense_weights is None:
            # ids.shape + (n_senses,)
            seqs = seqs.type(torch.LongTensor)
            sense_weight_logits = self.sense_weight_logits[seqs]
            sense_mask = self.sense_mask[seqs]
            sense_weights = masked_softmax(sense_weight_logits, sense_mask)
            
            # Checks
            check(seqs)
            check(sense_weight_logits)
            check(sense_mask)
            check(sense_weights)

        # TODO: Check weighted_sense_embeddings
        weighted_sense_embeddings = torch.sum(sense_embeddings * sense_weights.unsqueeze(-1), dim=-2)
        check(weighted_sense_embeddings)
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
        self.disambiguation_bert_config = BertConfig(
            hidden_size=self.embedding_size,
            num_hidden_layers=self.options.n_disambiguation_layers,
            intermediate_size=self.options.bert_intermediate_size,
            num_attention_heads=self.options.n_attention_heads,
            hidden_dropout_prob=self.options.dropout,
            attention_probs_dropout_prob=self.options.dropout,
            max_position_embeddings=self.max_seq_len,
        )

        self.prediction_bert_config = BertConfig(
            hidden_size=self.embedding_size,
            num_hidden_layers=self.options.n_prediction_layers,
            intermediate_size=self.options.bert_intermediate_size,
            num_attention_heads=self.options.n_attention_heads,
            hidden_dropout_prob=self.options.dropout,
            attention_probs_dropout_prob=self.options.dropout,
            max_position_embeddings=self.max_seq_len,
        )
    
        word_embeddings = self.make_word_embeddings(seqs)
        
        # Checks
        check(word_embeddings)
        check(torch.tensor(self.padding))

        if is_tf:
            word_embeddings = tf.convert_to_tensor(word_embeddings.detach().numpy())
            padding = tf.convert_to_tensor(self.padding)
        
            model = BertModel(
                config=self.disambiguation_bert_config,
                is_training=self.training,
                input_embeddings=word_embeddings,
                input_mask=padding,
            )
            
            reps = model.get_output() # (batch_size, sentence_len, embedding_size)
            reps = torch.from_numpy(reps.numpy())
        
        else:
            padding = torch.tensor(self.padding)
            model = BertModel(config=self.disambiguation_bert_config)
            
            reps = model.forward(
                is_training=self.training, 
                input_embeddings=word_embeddings, 
                input_mask=padding
            )

        # (batch_size, sentence_len, n_senses)
        sense_probs = self.calculate_sense_probs(seqs, reps) 
        # (batch_size, sentence_len, embedding_size)
        disambiguated_reps = self.make_word_embeddings(seqs, sense_weights=sense_probs)
    
        return disambiguated_reps, sense_probs

    def prediction_layer(self, reps):
        if is_tf:
            reps = tf.convert_to_tensor(reps.detach().numpy())
            padding = tf.convert_to_tensor(self.padding)
            model = BertModel(
                config=self.prediction_bert_config,
                is_training=self.training,
                input_embeddings=reps,
                input_mask=padding,
            )
            reps = model.get_output()
            reps = torch.from_numpy(reps.numpy())
        else: 
            padding = torch.tensor(self.padding)
            model = BertModel(self.prediction_bert_config)
            reps = model.forward(
                is_training=self.training, 
                input_embeddings=reps, 
                input_mask=padding
            )
        
        return reps
    
    def disambiguate(self, batch, method='prediction'):  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        unmasked_seqs = torch.tensor(batch.unmasked_seqs, dtype=torch.long, device=self.device)
        masked_seqs = torch.tensor(batch.masked_seqs, dtype=torch.long, device=self.device)
        target_positions = torch.tensor(batch.target_positions, dtype=torch.long, device=self.device)
        targets = torch.tensor(batch.targets, dtype=torch.long, device=self.device)
                
        padding = np.zeros(batch.unmasked_seqs.shape, dtype=np.int32)
        for i, l in enumerate(batch.seq_len):
            padding[i, :l] = 1
                                    
        dl_r = 0.5
        ml_coeff = 0.1

        # prob_tensors = {'prediction': self.qp}
        # if self.has_disambiguation_layer:
        #     prob_tensors['disambiguation'] = self.qd

        return self.run_forward(unmasked_seqs, masked_seqs, padding, target_positions, targets, dl_r, ml_coeff)
            
    # NOTE: update_smoothed_mean is used in _update_mean_qd and _update_mean_qp
    # NOTE: contextualize, disambiguate, get_target_probs add_get_mean_q, get_mean_sense_probs, get_sense_embeddings not called
    # NOTE: removed add_find_neighbors, geet_neighbors
    # NOTE: deduplicated_indexed_slices, average_gradients, and clip_gradients are only needed for manual gradient descent

    # def disambiguate(self, batch, method='prediction'):
    #     return self.disambiguate(batch, method=method)
    
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
        # self.models = torch.nn.ModuleList([
        #     PolyLMModel(vocab, self.n_senses, options, training=training).to('cpu')
        #     for _ in range(len(options.gpus.split(",")))  # Number of models to instantiate
        # ])
        model = PolyLMModel(self.vocab, self.n_senses, self.options, training=self.training)
        self.model = model

        ## Model setup across GPUs
        #self.models = torch.nn.ModuleList([
        #    PolyLMModel(vocab, self.n_senses, options, training=training).to(f'cuda:{i}')
        #    for i in range(len(options.gpus.split(",")))
        #])
        
        self.global_step = 0 # global_step = torch.ones(1, dtype=torch.int32)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=options.learning_rate)
  
    def train_model(self, corpus, num_epochs):
        # Initialize model and optimizer
        #model = PolyLMModel(self.vocab, self.n_senses, self.options, training=self.training)
        #self.model = model
        self.model.to(self.device)
        options = self.model.options
        optimizer = self.optimizer
        
        masking_policy = [float(x) for x in options.masking_policy.split()]
        batches = corpus.generate_batches(
            options.batch_size, 
            options.max_seq_len, 
            options.mask_prob,
            variable_length=True,
            masking_policy=masking_policy)
         
        # Add hooks for debugging
        def forward_hook(module, input, output):
            if torch.isnan(output).any():
                if debug: print(f"Forward hook: NaNs detected in {module}")

        def backward_hook(module, grad_input, grad_output):
            if any(torch.isnan(g).any() for g in grad_output):
                if debug: print(f"Backward hook: NaNs in gradients of {module}")
                

        for module in self.model.modules():
            module.register_forward_hook(forward_hook)
            module.register_backward_hook(backward_hook)
         
        # Training loop
        for epoch in range(num_epochs):
            total_loss = 0
            batch_count = 0
            for batch in batches:
                unmasked_seqs = torch.tensor(batch.unmasked_seqs, dtype=torch.long, device=self.device)
                masked_seqs = torch.tensor(batch.masked_seqs, dtype=torch.long, device=self.device)
                target_positions = torch.tensor(batch.target_positions, dtype=torch.long, device=self.device)
                targets = torch.tensor(batch.targets, dtype=torch.long, device=self.device)
                
                padding = np.zeros(batch.unmasked_seqs.shape, dtype=np.int32)
                for i, l in enumerate(batch.seq_len):
                    padding[i, :l] = 1
                                    
                # Forward pass
                optimizer.zero_grad()
                dl_r = 0.5
                ml_coeff = 0.1
                total_loss = self.model(unmasked_seqs, masked_seqs, padding, target_positions, targets, dl_r, ml_coeff)
                
                # Backward and optimize
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                                
                # Check for NaNs
                for name, param in self.model.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        if debug: print(f"NaNs in gradients of {name}")
                        
                # Loss accumulation
                total_loss += total_loss.item()
                batch_count += 1
            
                # Logging
                if batch_count > 0:
                    average_loss = total_loss / batch_count
                    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {average_loss:.4f}')
                else:
                    print("No batches processed.")

                # Saving:
                if (batch_count % options.save_every == 0):
                    print(self.model.state_dict())
                    torch.save(self.model.state_dict(), 'models/save_number' + str(batch_count))

        print('Finished Training')

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

    def disambiguate(self, batch, method='prediction'):
            return self.model.disambiguate(batch, method=method)