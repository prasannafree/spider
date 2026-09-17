import torch
import torch.nn as nn

from encoder.encoder import Encoder
from decoder.decoder import Decoder

class Generator(nn.Module):
    """
    Standard linear projection step.
    Maps from d_model (decoder output dimension) to vocab_size.
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns raw logits. 
        # Typically combined with nn.CrossEntropyLoss during training, 
        # which applies LogSoftmax internally.
        return self.proj(x)

class Transformer(nn.Module):
    """
    The full original Transformer architecture from "Attention Is All You Need".
    """
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_pad_idx: int,
        tgt_pad_idx: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 2048
    ):
        super().__init__()
        
        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            padding_idx=src_pad_idx,
            max_len=max_len
        )
        
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            padding_idx=tgt_pad_idx,
            max_len=max_len
        )
        
        self.generator = Generator(d_model, tgt_vocab_size)
        
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        
        self._init_params()
        
    def _init_params(self):
        """
        Initialize parameters using Glorot/Xavier uniform initialization.
        This is critical for Transformer convergence.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        Creates a boolean mask for the source sequence to ignore padding tokens.
        src shape: (batch_size, src_len)
        Returns: (batch_size, 1, 1, src_len)
        """
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
        
    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        Creates a mask for the target sequence that combines:
        1. Padding mask (ignore padding tokens)
        2. Causal mask (prevent looking into the future)
        tgt shape: (batch_size, tgt_len)
        Returns: (batch_size, 1, tgt_len, tgt_len)
        """
        tgt_pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)
        
        tgt_len = tgt.shape[1]
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        
        # Combine both masks
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        return tgt_mask

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        End-to-end forward pass for training.
        src shape: (batch_size, src_len)
        tgt shape: (batch_size, tgt_len)
        """
        # 1. Create masks
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        
        # 2. Pass through Encoder
        enc_out = self.encoder(src, mask=src_mask)
        
        # 3. Pass through Decoder
        dec_out = self.decoder(
            x=tgt,
            encoder_output=enc_out,
            tgt_mask=tgt_mask,
            src_mask=src_mask
        )
        
        # 4. Pass through Generator to get vocabulary logits
        logits = self.generator(dec_out)
        
        return logits
        
    # --- Inference Helpers ---
    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """Helper for autoregressive inference."""
        return self.encoder(src, mask=src_mask)
        
    def decode(
        self, 
        tgt: torch.Tensor, 
        encoder_output: torch.Tensor, 
        tgt_mask: torch.Tensor, 
        src_mask: torch.Tensor
    ) -> torch.Tensor:
        """Helper for autoregressive inference."""
        return self.decoder(tgt, encoder_output, tgt_mask, src_mask)
