import math 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from typing import Optional , Tuple



class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    Supports Self-Attention (Encoder/Decoder) and Cross-Attention (Decoder).
    """

    def __init__(self, d_model: int = 512 , num_heads:int = 8 , dropout: float = 0.1 )->None:
        super().__init__()
        self.d_model= d_model
        self.num_heads = num_heads
        self.dropout= dropout
        self.d_k = self.d_model // self.num_heads

        self.w_q = nn.Linear(d_model , d_model )
        self.w_k = nn.Linear(d_model , d_model )
        self.w_v = nn.Linear(d_model , d_model )
        self.w_o = nn.Linear(d_model , d_model )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor: 
        """
        q, k, v shapes: (batch_size, seq_len, d_model)
        mask shape: (batch_size, 1, 1, seq_len) or (batch_size, 1, seq_len, seq_len)
        """
        batch_size = q.size(0)
        
        query = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key   = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        

        attention_weights = F.softmax(scores , dim = -1)
        attention_weights = self.dropout(attention_weights)

        x = torch.matmul(attention_weights , value)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.w_o(x)





        

        

        
        

        
    
        

        
        