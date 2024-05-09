import copy
import json
import math
import six
import torch
import torch.nn as nn

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

    @classmethod
    def from_dict(cls, json_object):
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with tf.io.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class BertModel(nn.Module):
    def __init__(self, config):
        super(BertModel, self).__init__()
        #self.embeddings = BertEmbeddings(config)
        self.config = config        
        self.encoder = BertEncoder(config)

    def forward(self, is_training, input_embeddings, input_mask):
        assert not torch.isnan(input_embeddings).any(), "Tensor contains NaNs!"
        
        if not is_training:
            self.config.hidden_dropout_prob = 0.0
            self.config.attention_probs_dropout_prob = 0.0
            
        # Get embedding_output
        embedding_output = embedding_postprocessor(
            input_tensor=input_embeddings,
            use_position_embeddings=True,
            initializer_range=self.config.initializer_range,
            max_position_embeddings=self.config.max_position_embeddings,
            dropout_prob=self.config.hidden_dropout_prob)
        
        # Get attention_mask 
        attention_mask = input_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = attention_mask.to(dtype=torch.float32)
        attention_mask = (1.0 - attention_mask) * -10000.0
        encoder_outputs = self.encoder(embedding_output, attention_mask)
        return encoder_outputs

def embedding_postprocessor(input_tensor, 
                            use_position_embeddings=True,
                            initializer_range=0.02, 
                            max_position_embeddings=512, 
                            dropout_prob=0.1):

    batch_size, seq_length, width = input_tensor.size()
    output = input_tensor

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

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        
        self.linears = nn.ModuleList([self.query, self.key, self.value])

        # Initialize weights
        self._reset_parameters()

    def _reset_parameters(self):
        for linear in self.linears:
            linear.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q, k, v = [l(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
                   for l, x in zip(self.linears, (q, k, v))]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            #mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        p_attn = torch.nn.functional.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        x = torch.matmul(p_attn, v).transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.num_heads * self.d_k)

        return x

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(torch.nn.functional.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(num_heads, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.layernorm1(x)
        x = x + self.self_attn(x2, x2, x2, mask)
        x2 = self.layernorm2(x)
        x = x + self.feed_forward(x2)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
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
            self.intermediate_act_fn = torch.nn.functional.gelu
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