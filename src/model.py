import torch
import torch.nn as nn
import torch.optim as optim


import math
from random import *

import numpy as np
from utilities import get_attn_pad_mask

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))





class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, n_segments):
        super(Embedding, self).__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.segment_embedding = nn.Embedding(n_segments, d_model)

        self.normalization = nn.LayerNorm(d_model)

    def forward(self, tokens, segments):
        sequence_length = tokens.size(1)
        position_indices = torch.arange(sequence_length, dtype=torch.long)
        position_indices = position_indices.unsqueeze(0).expand_as(tokens)
        combined_embedding = self.token_embedding(tokens) + self.position_embedding(position_indices) + self.segment_embedding(segments)
        return self.normalization(combined_embedding)






class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.w_q = nn.Linear(d_model, d_k * n_heads)
        self.w_k = nn.Linear(d_model, d_k * n_heads)
        self.w_v = nn.Linear(d_model, d_v * n_heads)
        self.fc = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, attn_mask):
        residual = q
        batch_size = q.size(0)
        
        q_transformed = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_transformed = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_transformed = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        
        raw_scores = torch.matmul(q_transformed, k_transformed.transpose(-1, -2)) / np.sqrt(self.d_k)
        raw_scores.masked_fill_(attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1), -1e9)
        
        attention_scores = nn.Softmax(dim=-1)(raw_scores)
        context = torch.matmul(attention_scores, v_transformed)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.fc(context)
        
        output = self.layer_norm(output + residual)
        return output, attention_scores





class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, d_ff):
        super(EncoderLayer, self).__init__()

        self.enc_self_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, inputs, self_attn_mask):
        # Self-attention
        attention_output, attention_scores = self.enc_self_attn(inputs, inputs, inputs, self_attn_mask)
        
        # Feed-forward network
        output = self.pos_ffn(attention_output)
        
        return output, attention_scores

    



class BERT(nn.Module):
    def __init__(self, d_model, vocab_size, n_heads, n_segments, max_len, n_layers, d_k, d_v, d_ff):
        super(BERT, self).__init__()

        self.embedding_layer = Embedding(vocab_size, d_model, max_len, n_segments)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_k, d_v, d_ff) for _ in range(n_layers)])

        self.fc_transform = nn.Linear(d_model, d_model)
        self.tanh_activation = nn.Tanh()

        self.linear_transform = nn.Linear(d_model, d_model)
        self.gelu_activation = gelu
        self.layer_norm = nn.LayerNorm(d_model)
        self.classification_head = nn.Linear(d_model, 2)

        embed_weight = self.embedding_layer.token_embedding.weight
        n_vocab, n_dim = embed_weight.size()

        self.decoder_layer = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder_layer.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, input_ids, segment_ids, masked_pos):
        embedding_output = self.embedding_layer(input_ids, segment_ids)
        self_attention_mask = get_attn_pad_mask(input_ids, input_ids)

        for layer in self.encoder_layers:
            embedding_output, self_attention_scores = layer(embedding_output, self_attention_mask)

        # using the CLS token (0 index) for classification
        cls_output = self.tanh_activation(self.fc_transform(embedding_output[:, 0]))
        classification_logits = self.classification_head(cls_output)

        masked_pos_expanded = masked_pos[:, :, None].expand(-1, -1, embedding_output.size(-1))
        # expand masked positions for broadcasting

        masked_output = torch.gather(embedding_output, 1, masked_pos_expanded)

        masked_output = self.layer_norm(self.gelu_activation(self.linear_transform(masked_output)))
        language_model_logits = self.decoder_layer(masked_output) + self.decoder_bias
        
        return language_model_logits, classification_logits
