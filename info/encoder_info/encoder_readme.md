# `Encoder` Function Explanations & Examples

This document explains how the entire left side of the Transformer works!

---

## 1. `EncoderLayer(nn.Module)`

**Goal:** A single processing block that allows words in the SQL schema to look at each other and reason about their relationships.

### Example Execution
```python
layer = EncoderLayer(d_model=512, num_heads=8)
out = layer(x, mask)
```
Inside the `forward()` function, it performs the **Pre-LN** architecture:
1.  It Normalizes the input.
2.  It sends the input into `MultiHeadAttention` (Self-Attention, so Q, K, and V are all the exact same input tensor).
3.  It adds the output back to the original input (Residual Connection) to prevent the math from degrading in deep networks.
4.  It does the same thing for the `PositionwiseFeedForward` network.

---

## 2. `Encoder(nn.Module)`

**Goal:** The master controller. It takes raw text integers, embeds them, and loops them through 6 EncoderLayers to generate the final Context Vector.

### Example Execution
```python
# Raw integers (Batch size 1, Sequence length 3)
x = tensor([[2, 40, 165]]) 

encoder = Encoder(vocab_size=8000, num_layers=6)
context_vector = encoder(x, mask)
```
1.  `self.embeddings(x)` transforms the `[1, 3]` integers into a `[1, 3, 512]` grid of floats and adds the Sine/Cosine waves.
2.  The loop `for layer in self.layers:` pushes that `[1, 3, 512]` grid through the 1st EncoderLayer. The output goes into the 2nd layer, and so on, 6 times.
3.  The final `[1, 3, 512]` tensor is returned. It contains a deep, mathematical understanding of the entire database schema!
