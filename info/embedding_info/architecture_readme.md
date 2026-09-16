# Embeddings & Dataset Architecture

This document outlines the architectural guidelines and empirical design decisions for building the bridge between the BPE Tokenizer and the core Transformer model.

The code for this layer should be implemented in an `embeddings/` directory, separating it from raw data processing and core model logic.

## 1. Directory Structure

The proposed structure modularizes data loading, batch collation, and initial tensor embeddings:

```text
embeddings/
├── spider_dataset.py      # PyTorch Dataset (loads JSON, tokenizes, formats arrays)
├── collate.py             # DataLoader collate_fn (dynamic padding, mask generation)
└── static_embeddings.py   # Initial Transformer layers (Token + Positional Embeddings)
```

## 2. Sequence Length Limits (Empirical Analysis)

Before hardcoding maximum sequence lengths for truncation and padding, an analysis of the token length distributions on the entire Spider training set (8,659 examples) was conducted using our BPE tokenizer.

**Encoder Lengths (Question + Schema):**
*   **Mean**: 152.5
*   **95th Percentile**: 328
*   **99th Percentile**: 467
*   **Max**: 1,268

**Decoder Lengths (SQL Query):**
*   **Mean**: 25.2
*   **95th Percentile**: 57
*   **99th Percentile**: 74
*   **Max**: 118

**Design Decision:**
*   **`encoder_max_len = 512`**: Covers >99% of training examples. Only ~80 massive databases require truncation (saving enormous GPU memory over a 1024 limit).
*   **`decoder_max_len = 128`**: Covers 100% of targets while leaving a safe buffer for generation during inference.

---

## 3. Dataset Construction & Teacher Forcing

The `spider_dataset.py` is responsible for extracting the schema, appending the question, and building explicit token arrays.

### Schema Formatting
To preserve deterministic table-column relationships, the schema string should be formatted as:
`table1 (col1, col2) | table2 (col1, col2)`

### Explicit Token Array Construction
Do not rely on parsing `[SEP]` as a string. Instead, independently encode the components and construct the array using special token IDs:

```python
# Conceptual Array Construction
encoder_input_ids = [BOS] + question_ids + [SEP] + schema_ids + [EOS]
```

*Note on Truncation: If the array exceeds 512 tokens, it should be sliced *before* the `[EOS]` token to guarantee `[EOS]` is always the final token.*

### Teacher Forcing Layout
The dataset must output the following specific sequences to enable Teacher Forcing during training:
*   **Encoder Input**: `[BOS] Question [SEP] Schema [EOS]`
*   **Decoder Input**: `[BOS] SQL`
*   **Decoder Target**: `SQL [EOS]` (Used for loss calculation)

---

## 4. Batch Collation & Masking

The `collate_fn` should take a list of unpadded sequences from the Dataset and yield rectangular Tensors padded to the maximum length *within that specific batch*.

It must return the following tensors:
1.  `encoder_input_ids`
2.  `decoder_input_ids`
3.  `decoder_target_ids`
4.  `encoder_attention_mask` (1 = real token, 0 = padding `[PAD]`)
5.  `decoder_padding_mask` (1 = real token, 0 = padding `[PAD]`)

**Crucial Distinction**: The Collator only handles *padding* masks. The causal (look-ahead) mask for the Decoder should be implemented later within the core Transformer attention mechanism, not here.

---

## 5. Static Embeddings

The `static_embeddings.py` file should implement the initial layers that convert the integer batches into dense float tensors.

*   **`TokenEmbedding`**: Use `torch.nn.Embedding(vocab_size=8000, d_model=512, padding_idx=0)`. Note that `d_model=512` is highly recommended for standard Transformer sizing. This is a *learned* embedding from scratch, not a pre-trained Word2Vec/GloVe model.
*   **`PositionalEncoding`**: Implement standard Sine/Cosine positional encodings as a distinct, separate class for modularity.
*   **Forward Flow**: The input IDs should pass through the `TokenEmbedding`, be scaled by `sqrt(d_model)`, and then have the `PositionalEncoding` added to them. The final output tensor shape will be `[batch_size, seq_len, 512]`.
