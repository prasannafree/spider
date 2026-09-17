import torch
import torch.nn as nn
import copy
from typing import Optional

from embeddings.static_embeddings import TransformerEmbeddings
from decoder.decoder_layer import DecoderLayer

def get_clones(module, N):
    """
    Helper function to produce N identical layers.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Decoder(nn.Module):
    """
    The full Transformer Decoder module.
    Takes target token IDs, embeds them, and passes them through N Decoder layers.
    """
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int = 512, 
        num_layers: int = 6, 
        num_heads: int = 8, 
        d_ff: int = 2048, 
        dropout: float = 0.1, 
        padding_idx: int = 0,
        max_len: int = 2048
    ):
        super().__init__()
        
        # 1. Embeddings (shared implementation with Encoder)
        self.embeddings = TransformerEmbeddings(
            vocab_size=vocab_size, 
            d_model=d_model, 
            padding_idx=padding_idx, 
            max_len=max_len
        )
        
        # 2. Stack of N Decoder Layers
        single_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.layers = get_clones(single_layer, num_layers)
        
        # 3. Final LayerNorm (crucial for the Pre-LN architecture)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(
        self, 
        x: torch.Tensor, 
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x shape: (batch_size, seq_len) - Target token IDs
        encoder_output shape: (batch_size, src_seq_len, d_model) - Context from encoder
        tgt_mask shape: (batch_size, 1, seq_len, seq_len) - Causal + Padding mask
        src_mask shape: (batch_size, 1, 1, src_seq_len) - Encoder padding mask
        """
        seq_len = x.size(1)
        
        # 1. Convert integers to continuous embedding vectors + positional encoding
        x = self.embeddings(x)
        
        # 2. Pass through each of the layers sequentially
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
            
        # 3. Apply final LayerNorm (required for Pre-LN)
        x = self.norm(x)
        
        return x

