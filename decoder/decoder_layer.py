import torch
import torch.nn as nn
from typing import Optional

from attention.multi_head_attention import MultiHeadAttention
from attention.feed_forward import PositionwiseFeedForward

class DecoderLayer(nn.Module):
    """
    A single Pre-LN Decoder block.
    Architecture:
    1. LayerNorm -> Masked Self-Attention -> Dropout -> Add
    2. LayerNorm -> Cross-Attention (with Encoder) -> Dropout -> Add
    3. LayerNorm -> FeedForward -> Dropout -> Add
    """
    def __init__(self, d_model: int = 512, num_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # 3 LayerNorms because we have 3 sub-layers now!
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        encoder_output: torch.Tensor, 
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x: Target sequence (Decoder input)
        encoder_output: Context vector from the Encoder
        tgt_mask: Causal mask to prevent looking ahead in the target sequence
        src_mask: Padding mask from the source sequence
        """
        # --- 1. Masked Self-Attention Block ---
        norm_x = self.norm1(x)
        attn_out = self.self_attn(q=norm_x, k=norm_x, v=norm_x, mask=tgt_mask)
        x = x + self.dropout(attn_out)
        
        # --- 2. Cross-Attention Block ---
        # This is where the magic happens! Query comes from Decoder, Keys/Values from Encoder.
        norm_x = self.norm2(x)
        attn_out = self.cross_attn(q=norm_x, k=encoder_output, v=encoder_output, mask=src_mask)
        x = x + self.dropout(attn_out)
        
        # --- 3. Feed-Forward Block ---
        norm_x = self.norm3(x)
        ff_out = self.feed_forward(norm_x)
        x = x + self.dropout(ff_out)
        
        return x
