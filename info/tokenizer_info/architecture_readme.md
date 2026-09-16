# SpiderTokenizer Documentation

This document provides a comprehensive overview of the `SpiderTokenizer`, a custom Byte-Pair Encoding (BPE) tokenizer built specifically for a Text-to-SQL Encoder-Decoder Transformer on the Spider dataset.

## Table of Contents
1. [Architecture & Design Decisions](#architecture--design-decisions)
2. [Data Preparation & Leakage Prevention](#data-preparation--leakage-prevention)
3. [Special Tokens](#special-tokens)
4. [File Structure](#file-structure)
5. [Usage Guide](#usage-guide)
6. [Testing](#testing)

---

## Architecture & Design Decisions

The tokenizer uses the highly optimized HuggingFace `tokenizers` library (written in Rust) and is configured to handle the unique challenges of parsing both natural language (English) and structured language (SQL) simultaneously.

- **Algorithm**: Byte-Pair Encoding (BPE). BPE provides the best balance of vocabulary efficiency and handling of rare/unseen words.
- **Vocabulary Size**: `8,000`. This is large enough to capture common English words and SQL keywords efficiently, but small enough to force the model to learn subword components of rare database column names.
- **Case Preservation**: SQL is technically case-insensitive, but we **preserve case** (`NFC` normalization instead of lowercasing). This ensures the model learns the distinction between SQL keywords (often uppercase) and schema identifiers, and guarantees the decoder can reproduce exact casing.
- **Word Boundaries**: We use the `Metaspace` pre-tokenizer and decoder with explicit `▁` markers. This ensures that spaces are not lost when text is broken into subwords and re-assembled (a common issue with basic whitespace tokenizers).

---

## Data Preparation & Leakage Prevention

The tokenizer is trained on a custom corpus built from the Spider dataset. 
To prevent **data leakage** (where the tokenizer learns frequencies from validation/test sets), the corpus strictly uses:
1. `train_spider.json` (7,000 examples)
2. `train_others.json` (1,659 examples)
3. `tables.json` (Filtered strictly to the 146 databases that appear in the training sets above. Non-training schemas are ignored).

The corpus is extracted to `data/tokenizer_corpus.txt` line-by-line before training.

---

## Special Tokens

The tokenizer reserves 5 special tokens at fixed IDs (0-4) which are crucial for the Encoder-Decoder architecture:

| Token | ID | Purpose |
|-------|----|---------|
| `[PAD]` | 0 | **Padding**: Used to pad sequences to the same length in a batch. Placed at ID 0 for efficient masking in PyTorch. |
| `[UNK]` | 1 | **Unknown**: Used as a fallback when an input sequence contains characters that cannot be represented by the learned BPE vocabulary. |
| `[BOS]` | 2 | **Beginning of Sequence**: Marks the start of a sequence. Essential for the Decoder to know when to start generating SQL. |
| `[EOS]` | 3 | **End of Sequence**: Marks the end of a sequence. Signals the Decoder to stop generation. |
| `[SEP]` | 4 | **Separator**: Available to separate distinct contexts, e.g., `[BOS] DB Schema [SEP] Question [EOS]`. |

---

## File Structure

All code is contained within the `tokenizer/` directory:

* `prepare_data.py`: Extracts the training questions, SQL queries, and schema metadata into a plain text corpus.
* `train_tokenizer.py`: Defines the BPE architecture, sets up special tokens, and trains the model.
* `tokenizer.py`: Contains the `SpiderTokenizer` class. **This is the single interface you should use everywhere in your project.**
* `test_tokenizer.py`: A comprehensive suite of 59 tests to ensure correctness.
* `trained_tokenizer.json`: The actual saved weights and vocabulary of the trained model.

---

## Usage Guide

The `SpiderTokenizer` class wraps the raw HuggingFace tokenizer with a clean API tailored for Transformer training.

### Initialization
```python
from tokenizer import SpiderTokenizer

# Load the trained model
tok = SpiderTokenizer("tokenizer/trained_tokenizer.json")

print(f"Vocab size: {tok.vocab_size}") # 8000
print(f"Pad ID: {tok.pad_token_id}")   # 0
```

### Basic Encoding / Decoding
```python
text = "SELECT count(*) FROM department"

# Encode to raw IDs (No special tokens added)
ids = tok.encode(text) 

# Decode back to text (Spaces and casing are preserved)
decoded = tok.decode(ids)
```
*Note: Batch operations are available via `tok.encode_batch()` and `tok.decode_batch()`.*

### Model Training Formatting (Teacher Forcing)
When training an Encoder-Decoder, you need properly shifted inputs and targets. The `encode_for_model` method handles this automatically by applying the correct `[BOS]` and `[EOS]` framing.

```python
result = tok.encode_for_model(
    source="How many departments are there?",
    target="SELECT count(*) FROM department"
)

# Encoder gets: [BOS] source [EOS]
encoder_input = result["encoder_input"]

# Decoder gets teacher-forced input: [BOS] target
decoder_input = result["decoder_input"]

# Loss is calculated against shifted target: target [EOS]
decoder_target = result["decoder_target"]
```

### Inference Formatting
During generation, you only have the source question.
```python
result = tok.encode_for_model(source="How many departments are there?")
encoder_input = result["encoder_input"]

# Start generation by feeding [BOS] to the decoder...
```

---

## Testing

The module includes a robust validation suite (`test_tokenizer.py`) with 59 test cases. It verifies:
- **Roundtrip correctly**: `decode(encode(text)) == text`.
- **SQL Handling**: SQL Keywords (`SELECT`, `WHERE`), operators (`>=`, `!=`), and nested queries are correctly processed.
- **Model Readiness**: `encode_for_model` correctly shifts targets and applies `[BOS]`/`[EOS]`.
- **Edge cases**: Empty strings, skip_special_tokens toggles, etc.
