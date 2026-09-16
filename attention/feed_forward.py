import torch 
import math 
import torch.nn as nn 
import torch.nn.functional as F 


class PositionwiseFeedForward(nn.Module):
    """
    A simple two-layer fully-connected network applied to each position separately and identically.
    This provides non-linearity to the Transformer after the Attention mechanism.
    """

    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model , d_ff)
        self.w_2 = nn.Linear(d_ff , d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """        
        x shape : (batch_size , seq_len , d_model)        
        """
        x = F.relu(self.w_1(x))
        x= self.dropout(x)
        x = self.w_2(x)
        return x
        
        