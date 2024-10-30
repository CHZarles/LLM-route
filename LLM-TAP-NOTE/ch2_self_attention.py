import torch
from torch import nn


class MultiHeadAttentioin(nn.Module):
    def __init__(self, heads, d_model , dropout = 0.1):
        super().__init__()

        self.d_model = d_model
        self.d_model = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    
    # q: query ,k : key, v : value, d_k: dimension of key
    # transpose ： (batch_size, heads, seq_len, d_k)
    def attention(q, k ,v, d_k, mask=None , dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)

        return output
