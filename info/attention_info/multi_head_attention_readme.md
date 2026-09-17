# `MultiHeadAttention` Function Explanations & Examples

This document explains the core math behind the Multi-Head Attention layer.

---

## 1. `__init__(self, d_model, num_heads, dropout)`

**Goal:** Create the parameters required to project embeddings into Queries (Q), Keys (K), and Values (V).

### Example Execution
```python
attn = MultiHeadAttention(d_model=512, num_heads=8)
```
Instead of making 8 tiny linear layers, this creates one massive `Linear(512, 512)` for `w_q`, `w_k`, and `w_v`. This is an extreme hardware optimization (vectorization) so the GPU only has to run one massive matrix multiplication instead of 8 tiny ones.

---

## 2. `forward(self, q, k, v, mask)`

**Goal:** Perform the $Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$ formula.

### Example Execution
Imagine our sequence has 3 tokens (e.g. `[BOS] SELECT count`).
*   `q`, `k`, `v` shapes are `(1, 3, 512)`

**A. Splitting into Heads**
```python
query = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
```
The math transforms the `(1, 3, 512)` tensor into `(1, 8, 3, 64)`.
This essentially gives us 8 parallel brains (heads) that are each looking at the 3 words using 64 decimals.

**B. Scoring ($QK^T$)**
```python
scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(64)
```
For each head, it multiplies the Query matrix by the Key matrix. The result is a `(1, 8, 3, 3)` grid of scores. 
This grid represents how much Word 1 is paying attention to Word 1, Word 2, and Word 3. 

**C. Masking & Softmax**
```python
scores = scores.masked_fill(mask == 0, -1e9)
attention_weights = F.softmax(scores, dim=-1)
```
If we provided a mask, it turns the forbidden squares into `-1e9`. The `softmax` turns these scores into percentages (e.g., 90% attention on word 2, 10% on word 3). The `-1e9` values become exactly `0.0`.

**D. Multiplication by Value ($V$)**
```python
x = torch.matmul(attention_weights, value)
```
It multiplies the percentage weights by the actual Value embeddings. The result is put back together into `(1, 3, 512)` and returned!
