import torch
import torch.nn as nn
import math

import copy
import json

# TODO
# Add comments to the functions
# Indicate which tf segment the functions refer to

class BertConfig(object):
    def __init__(self, 
                 hidden_size=768, 
                 num_hidden_layers=12, 
                 num_attention_heads=12, 
                 intermediate_size=3072, 
                 hidden_act="gelu", 
                 hidden_dropout_prob=0.1, 
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512, 
                 initializer_range=0.02): 
        
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range

    # TODO: check this does the same as the six library
    @classmethod
    def from_dict(cls, json_object):
        config = BertConfig()
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    # TODO: check this does the same as tf.io.gfile.GFile
    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class BertModel(nn.Module):
    # TODO: check if necessary to add is_training, input_embeddings, and input_mask
    # parameters to the model init
    
    def __init__(self, config):
        super(BertModel, self).__init__()
        #self.embeddings = BertEmbeddings(config)
        self.config = config        
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(self, is_training, input_embeddings, input_mask=None, token_type_ids=None):
        # TODO: DON'T KNOW WHAT'S GOING ON HERE
        #if input_mask is None:
        #    input_mask = torch.ones_like(input_embeddings)

        #if token_type_ids is None:
        #    token_type_ids = torch.zeros_like(input_embeddings)

        # START HERE 
        # EMBEDDING_POSTPROCESSOR
        #embedding_output = self.embeddings(input_embeddings, token_type_ids)
        
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0
            
        # Get embedding_output
        embedding_output = embedding_postprocessor(
            input_tensor=input_embeddings,
            use_position_embeddings=True,
            #position_embedding_name="position_embeddings",
            initializer_range=self.config.initializer_range,
            max_position_embeddings=self.config.max_position_embeddings,
            dropout_prob=self.config.hidden_dropout_prob)
        
        
        # Get attention_mask
        batch_size, from_seq_length, _ = input_embeddings.shape # torch.Size([64, 16, 128])
        to_seq_length = input_mask.shape[1]
        to_mask = torch.reshape(input_mask, [batch_size, 1, to_seq_length])
        to_mask = to_mask.type(torch.float32)
        broadcast_ones = torch.ones([batch_size, from_seq_length, 1])
        attention_mask = broadcast_ones * to_mask
        
        #attention_mask = input_mask.unsqueeze(1).unsqueeze(2)
        #attention_mask = attention_mask.to(dtype=torch.float32)
        #attention_mask = (1.0 - attention_mask) * -10000.0
        
        encoder_outputs = self.encoder(embedding_output, attention_mask)
        sequence_output = encoder_outputs[-1]
        pooled_output = self.pooler(sequence_output)
        return sequence_output, pooled_output

# TODO: replace with torch gelu?
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def embedding_postprocessor(input_tensor, 
                            use_token_type=False,
                            token_type_ids=None, 
                            token_type_vocab_size=16,
                            #token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            #position_embedding_name="position_embeddings",
                            initializer_range=0.02, 
                            max_position_embeddings=512, 
                            dropout_prob=0.1):

    batch_size, seq_length, width = input_tensor.size()
    output = input_tensor

    if use_token_type:
        if token_type_ids is None:
            raise ValueError("`token_type_ids` must be specified if `use_token_type` is True.")
        
        token_type_embeddings = nn.Embedding(token_type_vocab_size, width)
        token_type_embeddings.weight.data.normal_(mean=0.0, std=initializer_range)
        token_type_embeddings = token_type_embeddings(token_type_ids)
        token_type_embeddings = torch.reshape(token_type_embeddings, [batch_size, seq_length, width])
        
        output += token_type_embeddings

    if use_position_embeddings:
        if seq_length > max_position_embeddings:
            raise ValueError("Sequence length is greater than the maximum allowed position embeddings.")
        
        position_embeddings = nn.Embedding(max_position_embeddings, width)
        position_embeddings.weight.data.normal_(mean=0.0, std=initializer_range)

        # TODO: Check this
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_tensor.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = position_embeddings(position_ids)
        output = output + position_embeddings

    # Layer normalization and dropout
    normalized_shape = (seq_length, width)
    output = torch.nn.functional.layer_norm(output, normalized_shape, eps=1e-12)
    
    if dropout_prob is not None and dropout_prob != 0.0:
        output = torch.nn.functional.dropout(output, p=dropout_prob)
    
    assert output.shape == input_tensor.shape
    
    return output

#class BertEmbeddings(nn.Module):
#    def __init__(self, config):
#        super(BertEmbeddings, self).__init__()
#        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
#        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
#        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)

#        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
#        self.dropout = nn.Dropout(config.hidden_dropout_prob)

#    def forward(self, input_embeddings, token_type_ids):
#        seq_length = input_embeddings.size(1)
#        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_embeddings.device)
#        position_ids = position_ids.unsqueeze(0).expand_as(input_embeddings)

#        words_embeddings = self.word_embeddings(input_embeddings)
#        position_embeddings = self.position_embeddings(position_ids)
#        token_type_embeddings = self.token_type_embeddings(token_type_ids)

#        embeddings = words_embeddings + position_embeddings + token_type_embeddings
#        embeddings = self.LayerNorm(embeddings)
#        embeddings = self.dropout(embeddings)
#        return embeddings

class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, input_mask):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, input_mask)
        return hidden_states

class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, input_mask):
        attention_output = self.attention(hidden_states, input_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states, input_mask):
        self_output = self.self(hidden_states, input_mask)
        attention_output = self.output(self_output, hidden_states)
        return attention_output

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, input_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + input_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if config.hidden_act == "gelu":
            self.intermediate_act_fn = gelu
        else:
            raise ValueError("Unsupported activation function: " + config.hidden_act)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # batch_size, sequence_length, hidden_size
        print("Shape of hidden_states before pooling:", hidden_states.shape)
        first_token_tensor = hidden_states    
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
