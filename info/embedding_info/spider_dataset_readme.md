# `SpiderDataset` Function Explanations & Examples

This document explains exactly how each function in `spider_dataset.py` operates, using a concrete example to trace the data flow.

---

## 1. `__init__(self, data_path, tables_path, tokenizer, encoder_max_len, decoder_max_len)`

**Goal:** Set up the dataset by loading the raw JSON files into memory and pre-processing the schemas so they are ready for fast lookup during training.

### Step-by-Step Execution
When you initialize the dataset:
```python
dataset = SpiderDataset(
    data_path="data/spider_data/train_spider.json",
    tables_path="data/spider_data/tables.json",
    tokenizer=my_tokenizer,
    encoder_max_len=512,
    decoder_max_len=128
)
```
1. It opens `train_spider.json` and loads it into a massive Python list of dictionaries called `self.examples`.
2. It opens `tables.json` and loads it into a list called `tables_raw`.
3. It immediately calls `_build_schemas()` to convert `tables_raw` into string representations.

---

## 2. `__len__(self)`

**Goal:** Tell PyTorch's DataLoader exactly how many examples exist in this dataset.

### Example
```python
def __len__(self) -> int:
    return len(self.examples)
```
If `train_spider.json` has exactly 7,000 examples, calling `len(dataset)` will return `7000`. PyTorch uses this to know when an epoch is over.

---

## 3. `_build_schemas(self, tables_raw)`

**Goal:** Take the raw JSON tables from Spider and deterministically format them into a string so the model can read them.

### The Input (`tables_raw`)
Imagine the `tables.json` file contains a very small database about a school:

```python
[
    {
        "db_id": "school_db",
        "table_names_original": ["student", "department"],
        "column_names_original": [
            [-1, "*"],          # The wildcard column (index -1)
            [0, "student_id"],  # Belongs to table 0 ("student")
            [0, "name"],        # Belongs to table 0 ("student")
            [1, "dept_id"],     # Belongs to table 1 ("department")
            [1, "budget"]       # Belongs to table 1 ("department")
        ]
    }
]
```

### Step-by-Step Execution inside the function

**A. Loop over the tables to group the columns:**
The function creates empty lists for each table index (`0` and `1`). 
It loops through the columns and assigns them. It ignores index `-1` (the `*` wildcard).
```python
table_to_cols = {
    0: ["student_id", "name"],
    1: ["dept_id", "budget"]
}
```

**B. Loop over `table_names_original` to build the string parts:**
*For Table `0` ("student"):*
- It grabs `["student_id", "name"]` and joins them with commas: `"student_id, name"`
- It formats it: `"student (student_id, name)"`

*For Table `1` ("department"):*
- It grabs `["dept_id", "budget"]` and joins them: `"dept_id, budget"`
- It formats it: `"department (dept_id, budget)"`

**C. Join all tables with the pipe `|` character:**
```python
schemas["school_db"] = "student (student_id, name) | department (dept_id, budget)"
```

### The Final Output
The function returns the final dictionary:
```python
{
    "school_db": "student (student_id, name) | department (dept_id, budget)"
}
```

---

## 4. `__getitem__(self, idx)`

**Goal:** Process a single training example from raw text into three strict integer arrays.

When PyTorch starts training, it asks the dataset for an item by its index, for example: `dataset[0]`.

### A. Extracting the Data
```python
example = self.examples[0]
```
The dataset pulls the 0th item:
*   `question`: `"Find students"`
*   `query`: `"SELECT name FROM student"`
*   `db_id`: `"school_db"`

### B. Grabbing the Schema & Adding Prefixes
It looks up `"school_db"` in the dictionary we built earlier and adds clear labels.
*   `q_text` becomes: `"Question: Find students"`
*   `s_text` becomes: `"Schema: student (id, name) | department (dept_id, budget)"`

### C. Independent Tokenization
We pass these plain strings to our BPE tokenizer. Let's pretend it gives us these arrays of integer IDs:
*   `q_ids` = `[10, 20, 22]` *(length: 3)*
*   `s_ids` = `[11, 30, 31, 32, 33, 34]` *(length: 6)*
*   `sql_ids` = `[40, 43, 42, 30]` *(length: 4)*

### D. The Encoder Truncation Logic
```python
bos = 2; eos = 3; sep = 4
fixed_len = 3 # We need space for [BOS], [SEP], [EOS]
```
The total length we need is: `3 (special) + 3 (question) + 6 (schema) = 12`. 
Since `12` is much less than our max limit (`512`), the truncation logic is skipped entirely!

We assemble the Encoder Input:
```python
encoder_input = [bos] + q_ids + [sep] + s_ids + [eos]
# encoder_input = [2, 10, 20, 22, 4, 11, 30, 31, 32, 33, 34, 3]
```

### E. The Decoder Formatting (Teacher Forcing)
The total SQL length is `4 (sql) + 1 (special) = 5`. 
Since `5` is much less than our limit (`128`), truncation is skipped.

We assemble the Teacher Forcing arrays:
```python
# Shifted Right (Starts with BOS, no EOS)
decoder_input = [bos] + sql_ids 
# decoder_input = [2, 40, 43, 42, 30]

# Shifted Left (Ends with EOS, no BOS)
decoder_target = sql_ids + [eos]
# decoder_target = [40, 43, 42, 30, 3]
```

### F. Returning the Result
The function finishes its job by returning a simple Python Dictionary containing those three arrays. This dictionary is then handed off to `collate.py`!
```python
return {
    "encoder_input_ids": encoder_input,
    "decoder_input_ids": decoder_input,
    "decoder_target_ids": decoder_target,
}
```
