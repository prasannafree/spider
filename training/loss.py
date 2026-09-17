import torch
import torch.nn as nn 

class LabelSmoothingLoss(nn.Module):
    """
    Standard CrossEntropyLoss with Label Smoothing.
    Instead of targeting [1.0, 0.0, 0.0], we target [0.9, 0.05, 0.05].
    This prevents the model from becoming overly confident and improves generalization.
    """
    def __init__(self,vocab_size:int ,padding_idx:int , smoothing:float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

        self.criterion = nn.KLDivLoss(reduction='sum')

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        x: Logits from the generator. Shape: (batch_size, seq_len, vocab_size)
        target: Target token IDs. Shape: (batch_size, seq_len)
        """
        # Ensure x is logged probabilities (required by KLDivLoss)
        x = x.log_softmax(dim=-1)
        
        # Flatten the batches and sequence lengths together
        # x shape becomes (batch_size * seq_len, vocab_size)
        x = x.contiguous().view(-1, self.vocab_size)
        # target shape becomes (batch_size * seq_len,)
        target = target.contiguous().view(-1)
        
        # Create a tensor of shape (batch_size * seq_len, vocab_size) filled with the smoothing value
        # We divide by (vocab_size - 2) because we don't smooth the correct word or the padding token
        true_dist = x.clone().fill_(self.smoothing / (self.vocab_size - 2))
        
        # Insert the confidence value (0.9) at the correct target indices
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        # Zero out the loss for padding tokens so the model isn't penalized for them
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
            
        # Calculate the KL Divergence between the model's guess (x) and our smoothed targets (true_dist)
        return self.criterion(x, true_dist)
    
        
    