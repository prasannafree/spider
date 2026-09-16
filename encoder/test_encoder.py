import torch
from encoder.encoder import Encoder

def test_encoder():
    print("=" * 60)
    print("Testing Encoder Architecture")
    print("=" * 60)
    
    # Configuration
    vocab_size = 8000
    d_model = 512
    num_heads = 8
    num_layers = 6
    batch_size = 4
    seq_len = 120
    
    print("1. Initializing Encoder (6 layers, Pre-LN)...")
    encoder = Encoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=0.0
    )
    
    # Create fake batch of token IDs
    print(f"2. Generating fake batch of shape [batch_size={batch_size}, seq_len={seq_len}]...")
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Create a fake mask (batch_size, 1, 1, seq_len)
    mask = torch.ones((batch_size, 1, 1, seq_len))
    
    print("3. Pushing data through the Encoder (Embeddings -> 6x Attention -> 6x FeedForward)...")
    try:
        output = encoder(x, mask)
        print("✓ Forward pass successful!")
        
        # Verify output shape
        expected_shape = (batch_size, seq_len, d_model)
        if output.shape == expected_shape:
            print(f"✓ Output shape is correct: {output.shape}")
        else:
            print(f"✗ Output shape is INCORRECT. Expected {expected_shape}, got {output.shape}")
            
    except Exception as e:
        print(f"✗ Forward pass FAILED with error:\n{e}")

    print("=" * 60)
    print("Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_encoder()
