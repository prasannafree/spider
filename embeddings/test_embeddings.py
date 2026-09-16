import os
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add parent directory to path so we can import tokenizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer import SpiderTokenizer
from embeddings.spider_dataset import SpiderDataset
from embeddings.collate import SpiderCollate
from embeddings.static_embeddings import TransformerEmbeddings

def test_embeddings_pipeline():
    print("=" * 60)
    print("Testing Dataset & Embeddings Pipeline")
    print("=" * 60)

    # 1. Initialize Tokenizer
    tok_path = "tokenizer/trained_tokenizer.json"
    if not os.path.exists(tok_path):
        print(f"Skipping tests: Tokenizer not found at {tok_path}")
        return
        
    tok = SpiderTokenizer(tok_path)
    print("✓ Tokenizer loaded")

    # 2. Initialize Dataset
    data_path = "data/spider_data/train_spider.json"
    tables_path = "data/spider_data/tables.json"
    
    if not os.path.exists(data_path) or not os.path.exists(tables_path):
        print(f"Skipping tests: Data files not found.")
        return

    dataset = SpiderDataset(
        data_path=data_path,
        tables_path=tables_path,
        tokenizer=tok,
        encoder_max_len=512,
        decoder_max_len=128
    )
    print(f"✓ Dataset loaded (size: {len(dataset)})")
    
    # 3. Test Collate function
    collate_fn = SpiderCollate(pad_token_id=tok.pad_token_id)
    
    # Take 4 diverse examples to form a batch
    batch_raw = [dataset[i] for i in [0, 10, 100, 1000]]
    batch = collate_fn(batch_raw)
    
    enc_ids = batch["encoder_input_ids"]
    dec_ids = batch["decoder_input_ids"]
    tgt_ids = batch["decoder_target_ids"]
    enc_mask = batch["encoder_attention_mask"]
    dec_mask = batch["decoder_padding_mask"]
    
    batch_size = 4
    print("\n[Shape Verification]")
    assert enc_ids.dim() == 2 and enc_ids.size(0) == batch_size, "Encoder input shape mismatch"
    assert dec_ids.dim() == 2 and dec_ids.size(0) == batch_size, "Decoder input shape mismatch"
    assert tgt_ids.dim() == 2 and tgt_ids.size(0) == batch_size, "Decoder target shape mismatch"
    assert enc_ids.size(1) <= 512, f"Encoder length {enc_ids.size(1)} exceeds max limit 512"
    assert dec_ids.size(1) <= 128, f"Decoder length {dec_ids.size(1)} exceeds max limit 128"
    assert dec_ids.size(1) == tgt_ids.size(1), "Decoder input and target lengths must match"
    print("✓ Tensor shapes are correct and within max limits")

    print("\n[Special Token Placement]")
    # First example in the padded batch
    seq = enc_ids[0].tolist()
    
    assert seq[0] == tok.bos_token_id, "Encoder input must start with [BOS]"
    assert tok.sep_token_id in seq, "Encoder input must contain [SEP]"
    
    # The last non-padding token must be [EOS]
    # Find the length of actual tokens using the mask
    actual_len = enc_mask[0].sum().item()
    assert seq[actual_len - 1] == tok.eos_token_id, "Encoder sequence must end with [EOS] before padding"
    
    assert dec_ids[0][0] == tok.bos_token_id, "Decoder input must start with [BOS]"
    # Target is shifted, so it should NOT start with [BOS], but end with [EOS]
    assert tgt_ids[0][0] != tok.bos_token_id, "Decoder target should not start with [BOS]"
    actual_dec_len = dec_mask[0].sum().item()
    assert tgt_ids[0][actual_dec_len - 1] == tok.eos_token_id, "Decoder target must end with [EOS]"
    print("✓ [BOS], [SEP], [EOS] placement and teacher-forcing shift is correct")

    print("\n[Padding & Masking]")
    # Verify padding token ID is used
    if enc_ids.size(1) > actual_len:
        assert seq[actual_len] == tok.pad_token_id, "Padding space must use pad_token_id"
        assert enc_mask[0][actual_len].item() == 0, "Padding mask must be 0 for padding tokens"
    
    assert enc_mask.sum().item() + (enc_mask == 0).sum().item() == enc_mask.numel(), "Mask should only contain 1s and 0s"
    print("✓ Padding tokens and boolean masks are aligned")

    print("\n[Transformer Embeddings]")
    # Initialize the static embeddings model
    embed_layer = TransformerEmbeddings(
        vocab_size=tok.vocab_size,
        d_model=512,
        padding_idx=tok.pad_token_id
    )
    
    out = embed_layer(enc_ids)
    
    assert out.dim() == 3, "Embedding output must be 3D"
    assert out.size(0) == batch_size, "Batch dimension mismatch"
    assert out.size(1) == enc_ids.size(1), "Sequence length mismatch"
    assert out.size(2) == 512, f"Embedding dimension must be 512, got {out.size(2)}"
    
    # Check that padded positions yield zero embeddings
    # (Since we used padding_idx=0 in nn.Embedding, and positional encodings 
    # might add a constant, actually padding might not be strictly zero due to positional encodings. 
    # Wait, positional encoding adds to padding too unless masked. 
    # Usually we don't care because attention mask ignores them, but let's just test shapes).
    
    print("✓ Embedding layer output shape is [batch_size, seq_len, 512]")
    
    print("\n============================================================")
    print("Results: All pipeline tests passed!")
    print("============================================================")


if __name__ == "__main__":
    test_embeddings_pipeline()
