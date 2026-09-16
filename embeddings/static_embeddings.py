import math
import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    """
    Learned token embeddings mapping integer token IDs to dense float vectors.
    Note: This is learned from scratch during Transformer training, 
    not a pre-trained Word2Vec/GloVe embedding.
    """
    def __init__(self, vocab_size: int, d_model: int, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=d_model, 
            padding_idx=padding_idx
        )
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard Transformer scaling: multiply by sqrt(d_model)
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding to inject sequence order 
    information into the transformer, since self-attention is permutation invariant.
    """
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        
        # Create a matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        
        # Create a vector of positions (0 to max_len - 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create the frequency denominator
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Register as a buffer so it's moved to the correct device but not updated by optimizer
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        # Add positional encoding up to the sequence length of x
        x = x + self.pe[:, :seq_len, :]
        return x

class TransformerEmbeddings(nn.Module):
    """
    Complete static embedding layer combining token embeddings 
    and positional encodings.
    """
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int = 512, 
        padding_idx: int = 0, 
        max_len: int = 2048
    ):
        super().__init__()
        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size, 
            d_model=d_model, 
            padding_idx=padding_idx
        )
        self.positional_encoding = PositionalEncoding(
            d_model=d_model, 
            max_len=max_len
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, seq_len) containing token IDs.
        Returns: (batch_size, seq_len, d_model) tensor.
        """
        out = self.token_embedding(x)
        out = self.positional_encoding(out)
        return out
