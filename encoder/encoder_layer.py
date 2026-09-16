import torch 
import torch.nn as nn 
import math 
import torch.nn.functional as F 

from typing import Optional 

from attention.multi_head_attention import MultiHeadAttention
from attention.feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int=512, d_ff: int=2048, num_heads: int=8, dropout: float=0.1):
        super().__init__()

        self.d_model = d_model          # dimension of the encoder model 
        self.d_ff = d_ff                # dimension of the feed forward layer
        self.num_heads = num_heads      # number of heads in the multi head attention
        self.dropout = dropout          # dropout rate

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x shape: (batch_size, seq_len, d_model)
        mask shape: (batch_size, 1, 1, seq_len)
        """
        # --- 1. Self-Attention Block (Pre-LN) ---
        norm_x = self.norm1(x)
        # Note: q=norm_x, k=norm_x, v=norm_x for self-attention
        attn_out = self.self_attn(norm_x, norm_x, norm_x, mask)
        
        # Apply dropout and add the residual connection (skip connection)
        x = x + self.dropout(attn_out)
        
        # --- 2. Feed-Forward Block (Pre-LN) ---
        norm_x = self.norm2(x)
        ff_out = self.feed_forward(norm_x)
        
        # Apply dropout and add the residual connection
        x = x + self.dropout(ff_out)
        
        return x
        
        