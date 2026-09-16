# `SpiderTokenizer` Function Explanations & Examples

This document explains exactly how each function in `tokenizer/tokenizer.py` operates to convert text into Subword Byte-Pair Encoding (BPE) IDs.

---

## 1. `__init__(self, tokenizer_path)`

**Goal:** Load the pre-trained HuggingFace tokenizer from a JSON file into memory.

### Example Execution
```python
tok = SpiderTokenizer("tokenizer/trained_tokenizer.json")
```
When you call this, it reads the JSON file which contains the entire learned vocabulary (e.g., mapping `"SELECT"` to `40`, and `" department"` to `439`). It stores this inside `self.tokenizer`.

---

## 2. `encode(self, text) -> List[int]`

**Goal:** Convert a single string of text into a list of integer IDs.

### Example Execution
```python
text = "SELECT name FROM student"
ids = tok.encode(text)
# Returns: [40, 165, 42, 30]
```
Behind the scenes, the BPE algorithm splits the text into words or subwords. Because SQL keywords are common, `"SELECT"` gets a single ID `40`. `"name"` gets `165`, `"FROM"` gets `42`, and `"student"` gets `30`.

---

## 3. `decode(self, ids, skip_special_tokens=False) -> str`

**Goal:** Convert a list of integer IDs back into a readable string.

### Example Execution
```python
ids = [40, 165, 42, 30]
text = tok.decode(ids)
# Returns: "SELECT name FROM student"
```
It looks up the string associated with each ID in its vocabulary and glues them back together. Because we use a `Metaspace` decoder, it correctly re-inserts the spaces between the words!

---

## 4. `encode_batch(self, texts) -> List[List[int]]`

**Goal:** Encode multiple strings simultaneously using Rust multithreading for extreme speed.

### Example Execution
```python
texts = [
    "SELECT name FROM student", 
    "How many departments?"
]
batch_ids = tok.encode_batch(texts)
# Returns: [
#    [40, 165, 42, 30], 
#    [20, 21, 439, 31]
# ]
```

---

## 5. `decode_batch(self, ids_list) -> List[str]`

**Goal:** Decode multiple arrays of IDs simultaneously.

### Example Execution
```python
ids_list = [
    [40, 165, 42, 30], 
    [20, 21, 439, 31]
]
texts = tok.decode_batch(ids_list)
# Returns: ["SELECT name FROM student", "How many departments?"]
```

---

## 6. `encode_for_model(self, source, target) -> Dict`

**Goal:** A convenience function to encode an entire training pair and wrap them in the special `[BOS]` and `[EOS]` tokens. *(Note: Our Dataset ended up doing this manually for finer control over truncation, but this function is useful for quick inference testing).*

### Example Execution
```python
source = "Find students"
target = "SELECT name FROM student"

result = tok.encode_for_model(source, target)
```
**Output Dictionary:**
*   `encoder_input`: `[2, 201, 22, 3]` *(Starts with BOS `2`, ends with EOS `3`)*
*   `decoder_input`: `[2, 40, 165, 42, 30]` *(Starts with BOS `2`)*
*   `decoder_target`: `[40, 165, 42, 30, 3]` *(Ends with EOS `3`)*

---

## 7. Properties (`vocab_size`, `pad_token_id`, etc.)

**Goal:** Provide direct access to the special token IDs without having to hardcode them everywhere.

### Example Execution
```python
print(tok.vocab_size)   # Output: 8000
print(tok.pad_token_id) # Output: 0
print(tok.bos_token_id) # Output: 2
print(tok.eos_token_id) # Output: 3
print(tok.sep_token_id) # Output: 4
```
This guarantees that if you ever retrain the tokenizer and the IDs change, the rest of your PyTorch code will automatically adapt!
