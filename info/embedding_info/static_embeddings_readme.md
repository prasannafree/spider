# `Static Embeddings` Function Explanations & Examples

This document explains exactly how each class and function in `static_embeddings.py` operates to convert integer tokens into continuous mathematics.

---

## 1. `TokenEmbedding(nn.Module)`

**Goal:** Convert 1D integer IDs into 2D continuous float vectors.

### `__init__(self, vocab_size, d_model, padding_idx)`
When initialized, it creates a massive PyTorch lookup table (`nn.Embedding`) with 8,000 rows (one for each word in our vocabulary) and 512 columns (the embedding dimension). It also stores the `d_model` size for scaling later.
```python
# Setup: 8000 words, 512 numbers per word, pad token is 0
embed = TokenEmbedding(vocab_size=8000, d_model=512, padding_idx=0)
```

### `forward(self, x)`
**The Input:**
Receives a 2D Tensor from the Collator containing integer IDs. 
Let's pretend our input sequence is just two tokens long: `[BOS]` (ID `2`) and `"SELECT"` (ID `40`).
```python
x = tensor([[2, 40]]) # Shape: [batch_size=1, seq_len=2]
```

**A. The Lookup (`self.embedding(x)`)**
It literally replaces the integer `2` with the exact 512 floating-point numbers stored in row `2` of its matrix.
```python
# Before: 
[2]

# After Lookup: 
[0.05, -1.2, 3.14, 0.002, ... 512th number]
```

**B. The Scaling (`* math.sqrt(self.d_model)`)**
It multiplies these vectors by `√512` (roughly `22.62`) so that positional encodings don't overpower the word meanings.
```python
# After scaling:
[1.13, -27.14, 71.02, 0.045, ... 512th number]
```

---

## 2. `PositionalEncoding(nn.Module)`

**Goal:** Inject the concept of "time" or "order" into the tokens, because Transformers have no idea what order words appear in.

### `__init__(self, d_model, max_len)`
When the model is first created, it pre-computes a massive grid of Sine and Cosine waves up to `max_len` (e.g., 2048 positions). 
*   The 1st word gets a specific signature of sine/cosine values.
*   The 2nd word gets a slightly shifted signature.

```python
# Pre-computed Position 1 Signature:
[0.0, 1.0, 0.0, 1.0, ... ] # 512 numbers

# Pre-computed Position 2 Signature:
[0.84, 0.54, 0.02, 0.99, ... ] # 512 numbers
```
It stores this giant matrix in `self.pe` so it never has to calculate sine/cosine waves during training!

### `forward(self, x)`
**The Input:**
Receives the `[1, 2, 512]` float tensor from the `TokenEmbedding` step.

**The Addition:**
It slices the pre-computed `pe` matrix to match the current sequence length, and adds it directly on top of our token decimals.
```python
# Word 1 ([BOS]) + Position 1 Signature:
  [1.13, -27.14, 71.02, ...] 
+ [0.00,   1.00,  0.00, ...]
= [1.13, -26.14, 71.02, ...]

# Word 2 ("SELECT") + Position 2 Signature:
  [ ... 512 scaled floats ... ]
+ [ ... 512 position floats . ]
= [ ... 512 combined floats . ]
```

---

## 3. `TransformerEmbeddings(nn.Module)`

**Goal:** Act as the single "wrapper" that runs both of the steps above in sequence.

### `__init__(self, vocab_size, d_model, padding_idx, max_len)`
It initializes both the `TokenEmbedding` and `PositionalEncoding` modules and holds them as sub-layers.

### `forward(self, x)`
This is the exact function our Neural Network will call!
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    out = self.token_embedding(x)
    out = self.positional_encoding(out)
    return out
```

*   **Input:** `x` (Shape: `[batch_size, seq_len]`)
*   **Output:** `out` (Shape: `[batch_size, seq_len, 512]`) 

The resulting tensor contains both the semantic meaning of the words (from TokenEmbedding) and the exact order they appeared in the sentence (from PositionalEncoding). It is ready for Multi-Head Attention!
