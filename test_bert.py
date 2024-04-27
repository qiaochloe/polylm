import torch
from bert import BertModel, BertConfig

def test_bert():
    # Configuration for BERT
    config = BertConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        vocab_size=30522  # Size of your vocabulary
    )

    # Initialize BERT model
    model = BertModel(config)

    # Dummy inputs
    input_ids = torch.tensor([[31, 51, 99], [15, 5, 1]])  # Batch size 1, sequence length 10
    attention_mask = torch.tensor([[1, 1, 1], [1, 1, 1]])  # All inputs are non-padded

    # Forward pass through BERT
    sequence_output, pooled_output = model(input_ids, attention_mask)

    print("Sequence Output Shape:", sequence_output.shape)  # Expected shape: (1, 10, 768)
    print("Pooled Output Shape:", pooled_output.shape)  # Expected shape: (1, 768)

if __name__ == "__main__":
    test_bert()