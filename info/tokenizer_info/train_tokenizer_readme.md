# `Train Tokenizer` Function Explanations & Examples

This document explains exactly how the `train_spider_tokenizer` function in `tokenizer/train_tokenizer.py` operates to build our custom BPE model.

---

## 1. `train_spider_tokenizer(corpus_path, save_path, vocab_size)`

**Goal:** Read the raw text corpus, mathematically learn the most common character combinations (BPE), inject our special tags, and save the final vocabulary to a file.

### Step-by-Step Execution

**A. Initialization**
```python
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
```
It starts with a blank HuggingFace Byte-Pair Encoding (BPE) model. BPE starts by treating every single letter as a token (e.g., `'S'`, `'E'`, `'L'`, `'E'`, `'C'`, `'T'`). It will eventually learn to merge them into words.

**B. Normalizer & Pre-tokenizer**
```python
tokenizer.normalizer = normalizers.Sequence([
    normalizers.Replace(Regex(r"\s+"), " "), 
])
tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
```
It cleans up the text by removing double spaces. The `Metaspace` algorithm replaces spaces with a special `_` character so that when we decode the tokens later, the spaces are preserved perfectly without us having to write messy string-joining logic.

**C. The Training Loop**
```python
trainer = BpeTrainer(
    vocab_size=8000,
    special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]"]
)
tokenizer.train([corpus_path], trainer)
```
This is where the magic happens. It reads the giant `spider_corpus.txt` file and looks for the most common adjacent characters.
*   It notices `'S'` and `'E'` happen together often, and merges them to `'SE'`.
*   It notices `'SE'` and `'L'` happen together often, and merges them to `'SEL'`.
*   It keeps merging until it finds the 8,000 most common chunks of text (which will end up being full words like `"SELECT"`, `"FROM"`, `"department"`, etc.).
*   It strictly reserves IDs 0 through 4 for our special formatting tokens (`[PAD]`, `[UNK]`, etc.).

**D. The Decoder**
```python
tokenizer.decoder = decoders.Metaspace()
```
It attaches the reverse of the `Metaspace` pre-tokenizer so that the `decode()` function knows how to turn the `_` characters back into normal spaces.

**E. Saving**
```python
tokenizer.save(save_path)
```
It saves the entire learned dictionary to `tokenizer/trained_tokenizer.json` so we never have to spend time training it again.

### Example of the Final Learned JSON File
If you were to open the `trained_tokenizer.json` file on your hard drive, it would look something like this:
```json
{
  "model": {
    "type": "BPE",
    "vocab": {
      "[PAD]": 0,
      "[UNK]": 1,
      "[BOS]": 2,
      "[EOS]": 3,
      "[SEP]": 4,
      "S": 5,
      "E": 6,
      "L": 7,
      "C": 8,
      "T": 9,
      "SELECT": 40,
      "FROM": 42,
      "department": 439
    }
  }
}
```
Now, whenever we call `tokenizer.encode("SELECT")`, it instantly looks at this JSON file and returns `40`!
