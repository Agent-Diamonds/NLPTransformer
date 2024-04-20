import torch
import torch.nn as nn 
import torch.nn.functional as F
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


class MultiHeadedAttention(nn.Module):
    
    def __init__(self, num_attn_heads, hidden_size, dropout_prob):
        super(MultiHeadedAttention, self).__init__()
        
        
        self.d_k = hidden_size // num_attn_heads
        self.num_attn_heads = num_attn_heads
        self.dropout = nn.Dropout(p=dropout_prob)
        
        #init the weights
        self.query = torch.nn.Linear(hidden_size, hidden_size) #Q
        self.key = torch.nn.Linear(hidden_size, hidden_size) #K
        self.value = torch.nn.Linear(hidden_size, hidden_size)#V
        self.W = torch.nn.Linear(hidden_size, hidden_size) #W

    
    def forward(self, query, key, value, attention_mask):
        expanded_attention_mask = self._expand_attention_mask(attention_mask)
    
        query, key, value = self._linear_transform(query, key, value)
        query, key, value = self._reshape_and_permute(query, key, value)
    
        scores = self._scaled_dot_product_attention(query, key, expanded_attention_mask)
        weights = self._apply_softmax_and_dropout(scores)
    
        context = self._weighted_sum(weights, value)
        context = self._reshape_and_permute_back(context)
        return self.W(context)

    def _expand_attention_mask(self, attention_mask):
        expanded_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        return expanded_attention_mask.repeat(1, self.num_attn_heads, 1, 1)

    def _linear_transform(self, query, key, value):
        query, key, value = self.query(query), self.key(key), self.value(value)
        return query, key, value

    def _reshape_and_permute(self, query, key, value):
        query = query.view(query.shape[0], -1, self.num_attn_heads, self.d_k).permute(0, 2, 1, 3)
        key = key.view(key.shape[0], -1, self.num_attn_heads, self.d_k).permute(0, 2, 1, 3)
        value = value.view(value.shape[0], -1, self.num_attn_heads, self.d_k).permute(0, 2, 1, 3)
        return query, key, value

    def _scaled_dot_product_attention(self, query, key, expanded_attention_mask):
        scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / math.sqrt(query.size(-1))
        scores = scores.masked_fill(expanded_attention_mask == 0, 1e-9)
        return scores

    def _apply_softmax_and_dropout(self, scores):
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        return weights

    def _weighted_sum(self, weights, value):
        context = torch.matmul(weights, value)
        return context

    def _reshape_and_permute_back(self, context):
        context = context.permute(0, 2, 1, 3).contiguous().view(context.shape[0], -1, self.num_attn_heads * self.d_k)
        return context


class PositionWiseFeedForward(nn.Module):
    def __init__(self, hidden_size, dropout_prob):
        super(PositionWiseFeedForward, self).__init__()
        middle_size = 4 * hidden_size #per devlin
        self.fc1 = nn.Linear(hidden_size, middle_size)
        self.fc2 = nn.Linear(middle_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.activation = nn.GELU()

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.fc2(self.dropout(out))
        return out


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_attn_heads, dropout_prob):
        super(EncoderLayer, self).__init__()
        self.layernorm = nn.LayerNorm(hidden_size) #normalizing
        self.self_multihead = MultiHeadedAttention(num_attn_heads, hidden_size, dropout_prob)
        self.feed_forward = PositionWiseFeedForward(hidden_size, dropout_prob)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, embeddings, attention_mask):
        #print(embeddings)
        #print(mask)
        attention = self.dropout(self.self_multihead(embeddings, embeddings, embeddings, attention_mask))
        # residual layer
        residual = self.layernorm(attention + embeddings)#residual connection and normalize after attention
        feed_forward_out = self.dropout(self.feed_forward(residual))
        encoded = self.layernorm(feed_forward_out + residual)#residual connection and normalize after ffnn
        return encoded




