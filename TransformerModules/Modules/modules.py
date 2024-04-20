import torch
import torch.nn as nn 
import math

class JointEmbedding(nn.Module):

    def __init__(self, vocab_size, hidden_size, seq_len, dropout_prob, device):
        super(JointEmbedding, self).__init__()
        self.segment_embedding = nn.Embedding(num_embeddings=3, embedding_dim=hidden_size, padding_idx=0)
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size, padding_idx=0)
        self.dropout = nn.Dropout(p=dropout_prob)
        # Calculate positional embeddings during initialization
        self.position = self.calculate_positional_embeddings(hidden_size, seq_len, device)

    def forward(self, sequence, segment_label):
        # Combine embeddings
        #print("Embedding token type id")
        #print(segment_label)
        #print(sequence)
        x = self.token_embedding(sequence) + self.position + self.segment_embedding(segment_label)
        return self.dropout(x)

    def calculate_positional_embeddings(self, d_model, seq_length, device):
        scalar = 10000
        pos_emb = torch.zeros(seq_length, d_model).float().to(device)
        pos_emb.require_grad = False  # Not optimizing as a parameter

        for pos in range(seq_length):
            for i in range(0, d_model, 2):
                pos_emb[pos, i] = math.sin(pos / (scalar ** ((2 * i) / d_model)))
                pos_emb[pos, i + 1] = math.cos(pos / (scalar ** ((2 * (i + 1)) / d_model)))
        pos_emb = pos_emb.unsqueeze(0)
        return pos_emb


