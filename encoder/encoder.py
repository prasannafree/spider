import torch
import torch.nn as nn
import copy
from typing import Optional

from embeddings.static_embeddings import TransformerEmbeddings
from encoder.encoder_layer import EncoderLayer

def get_clones(module, N):
    """
    Helper function to produce N identical layers.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    """
    The full Transformer Encoder module.
    Takes raw token IDs, embeds them, and passes them through N Encoder layers.
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
        
        # 1. Embeddings (from our previous module!)
        self.embeddings = TransformerEmbeddings(
            vocab_size=vocab_size, 
            d_model=d_model, 
            padding_idx=padding_idx, 
            max_len=max_len
        )
        
        # 2. Stack of N Encoder Layers
        single_layer = EncoderLayer(d_model, d_ff, num_heads, dropout)
        self.layers = get_clones(single_layer, num_layers)
        
        # 3. Final LayerNorm (crucial for the Pre-LN architecture)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x shape: (batch_size, seq_len) - Raw integer IDs
        mask shape: (batch_size, 1, 1, seq_len) - Boolean padding mask
        """
        # 1. Convert integers to continuous embedding vectors + positional encoding
        x = self.embeddings(x)
        
        # 2. Pass through each of the 6 layers sequentially
        for layer in self.layers:
            x = layer(x, mask)
            
        # 3. Apply final LayerNorm (required because we used Pre-LN)
        # This final tensor is the "Context Vector" that will be sent to the Decoder!
        return self.norm(x)
