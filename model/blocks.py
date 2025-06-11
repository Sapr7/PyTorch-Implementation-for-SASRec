import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        scale = -(math.log(10000.0) / d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * scale)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SASRecBlock(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = FeedForward(d_model, hidden_dim, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, padding_mask=None):
        # padding_mask: [B, T], True means PAD (ignored by attention)
        x_ln = self.ln1(x)
        attn_out, _ = self.attn(
            x_ln,
            x_ln,
            x_ln,
            key_padding_mask=padding_mask,
        )
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x
