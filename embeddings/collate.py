from typing import Dict, List
import torch

class SpiderCollate:
    """
    DataLoader collate function for Spider dataset batches.
    
    Responsible for:
    1. Finding the maximum sequence length dynamically within a specific batch.
    2. Padding all sequences in the batch to that maximum length.
    3. Generating padding masks (1 for real tokens, 0 for padding).
    
    Note: This generates *padding* masks for the decoder, not causal masks. 
    Causal (look-ahead) masks should be applied inside the Transformer attention 
    layers later. Furthermore, some PyTorch attention modules expect padding masks 
    to be boolean tensors where True = ignore/padding. These integer masks (1=real, 
    0=pad) should be converted appropriately when passed to those modules.
    """
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        # Find maximum lengths in this specific batch
        max_enc_len = max(len(ex["encoder_input_ids"]) for ex in batch)
        max_dec_len = max(len(ex["decoder_input_ids"]) for ex in batch)

        encoder_input_ids = []
        decoder_input_ids = []
        decoder_target_ids = []
        
        encoder_attention_mask = []
        decoder_padding_mask = []

        for ex in batch:
            # Encoder padding
            enc_ids = ex["encoder_input_ids"]
            enc_pad_len = max_enc_len - len(enc_ids)
            encoder_input_ids.append(enc_ids + [self.pad_token_id] * enc_pad_len)
            encoder_attention_mask.append([1] * len(enc_ids) + [0] * enc_pad_len)

            # Decoder padding (input and target are exactly the same length)
            dec_in_ids = ex["decoder_input_ids"]
            dec_tgt_ids = ex["decoder_target_ids"]
            dec_pad_len = max_dec_len - len(dec_in_ids)
            
            decoder_input_ids.append(dec_in_ids + [self.pad_token_id] * dec_pad_len)
            decoder_target_ids.append(dec_tgt_ids + [self.pad_token_id] * dec_pad_len)
            decoder_padding_mask.append([1] * len(dec_in_ids) + [0] * dec_pad_len)

        # Convert to PyTorch tensors (Long type for embedding lookups)
        return {
            "encoder_input_ids": torch.tensor(encoder_input_ids, dtype=torch.long),
            "decoder_input_ids": torch.tensor(decoder_input_ids, dtype=torch.long),
            "decoder_target_ids": torch.tensor(decoder_target_ids, dtype=torch.long),
            "encoder_attention_mask": torch.tensor(encoder_attention_mask, dtype=torch.long),
            "decoder_padding_mask": torch.tensor(decoder_padding_mask, dtype=torch.long),
        }
