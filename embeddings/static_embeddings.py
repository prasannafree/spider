import math 
import torch 
import torch.nn as nn 


class TokenEmbedding(nn.Module):
    def __init__(self,vocab_size :int ,d_model:int , padding_idx:int=0):
        super().__init__()
        self.embeddings = nn.Embedding(num_embeddings = vocab_size, embedding_dim = d_model, padding_idx = padding_idx)
        self.d_model = d_model
    
    def forward(self,x:torch.Tensor) -> torch.Tensor:          # x shape: (batch_size, seq_len)
        return self.embeddings(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self,d_model:int , max_seq_len: int = 2048):
        super().__init__()
        self.d_model = d_model              # embedding dimension 
        self.max_seq_len = max_seq_len      # max number of token allowed = context window of encoder 

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices 
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices 
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self,x: torch.Tensor) -> torch.Tensor :   # x shape: (batch_size, seq_len, d_model) 
        x = x + self.pe[:,:x.size(1),:]
        return x


class TransformerEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, padding_idx: int = 0, max_len: int = 2048):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size=vocab_size , d_model=d_model , padding_idx=padding_idx)
        self.positional_embedding = PositionalEncoding(d_model=d_model, max_seq_len=max_len)
        self.max_len = max_len
    
    def forward(self , x:torch.Tensor) -> torch.Tensor :
        out = self.token_embedding(x)
        out = self.positional_embedding(out)
        return out
        

        


        


