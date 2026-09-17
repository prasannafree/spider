# `Decoder` Function Explanations & Examples

This document explains how the right side of the Transformer works, generating SQL word by word!

---

## 1. `DecoderLayer(nn.Module)`

**Goal:** A processing block that looks at the SQL query written so far, and compares it to the Encoder's Context Vector to figure out what to write next.

### Example Execution
Inside the `forward()` function, it performs three blocks:
1.  **Masked Self-Attention:** It looks at the SQL words generated so far. We provide a Causal Mask here so that the network is mathematically blinded to future words (e.g., word 1 cannot pay attention to word 2).
2.  **Cross-Attention:** The Query (`q`) comes from the Masked Self-Attention. However, the Key (`k`) and Value (`v`) come directly from the `encoder_output`! This allows the SQL query to "search" the database schema to figure out what table name or column name to type next.
3.  **Feed-Forward:** It adds non-linear reasoning.

---

## 2. `Decoder(nn.Module)`

**Goal:** The master controller. It takes the target SQL tokens, embeds them, loops them through 6 DecoderLayers, and guesses the final word!

### Example Execution
```python
decoder = Decoder(vocab_size=8000)
# Returns a shape of (batch_size, seq_len, 8000)
predictions = decoder(x, encoder_output)
```

**Missing Components Note:**
To make this class function properly as the final output layer of a Transformer, it requires two special pieces:
1.  **Causal Mask Generator:** It needs a helper function to automatically generate the diagonal triangle of `1`s and `0`s to prevent the model from cheating during training.
2.  **The Generator:** After passing through all 6 DecoderLayers, the output is a `(batch, seq, 512)` tensor. We need a massive final linear layer `nn.Linear(512, 8000)` to project those 512 decimals up to 8,000 decimals so we can guess the exact word from the Tokenizer's vocabulary!
