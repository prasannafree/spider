# `Tokenizer Data Preparation` Function Explanations & Examples

This document explains exactly how each function in `tokenizer/prepare_data.py` operates to extract the raw text from the Spider JSONs so the Tokenizer can learn its vocabulary.

---

## 1. `load_json(file_path)`

**Goal:** Safely load a JSON file from disk into a Python list/dictionary.

### Example Execution
```python
data = load_json("data/spider_data/train_spider.json")
```
It reads the file, parses the JSON format, and returns the raw Python structures. It also contains error handling so if the file doesn't exist, it prints a clean error instead of crashing mysteriously.

---

## 2. `collect_db_ids(spider_data)`

**Goal:** Find all the unique databases that appear in our training set.

### The Input (`spider_data`)
Imagine our training set has just three questions:
```python
[
    {"db_id": "school_db", "question": "..."},
    {"db_id": "hospital_db", "question": "..."},
    {"db_id": "school_db", "question": "..."}
]
```

### Example Execution
```python
db_ids = collect_db_ids(spider_data)
# Returns: {"school_db", "hospital_db"}
```
It loops through all examples and uses a Python `set` to guarantee that we only get a unique list of databases, ignoring duplicates.

---

## 3. `extract_schema_metadata(tables_data, valid_db_ids)`

**Goal:** Extract the table names and column names, but *only* for the databases we collected in the previous step (preventing Data Leakage from the dev/test sets).

### The Input
```python
tables_data = [
    {"db_id": "school_db", "table_names_original": ["student"], "column_names_original": [[0, "name"]]},
    {"db_id": "secret_test_db", "table_names_original": ["aliens"], "column_names_original": [[0, "ufo"]]}
]
valid_db_ids = {"school_db"}
```

### Example Execution
```python
metadata_lines = extract_schema_metadata(tables_data, valid_db_ids)
# Returns: ["student", "name"]
```
It loops through all databases in `tables.json`. If it hits `"secret_test_db"`, it skips it completely! If it hits `"school_db"`, it extracts the table name `"student"` and column name `"name"` and adds them to our massive list of text.

---

## 4. `prepare_corpus(spider_path, tables_path, output_path)`

**Goal:** The main wrapper function that executes all the steps above and saves the final result to a giant text file.

### Step-by-Step Execution
1. Loads `train_spider.json` and `tables.json`.
2. Extracts every single natural language Question and SQL Query from `train_spider.json`.
3. Calls `collect_db_ids` and `extract_schema_metadata` to safely get all schema words.
4. Glues the Questions, SQL Queries, and Schema Metadata into one massive array of strings.
5. Writes them all to `tokenizer/spider_corpus.txt`.

### The Final Output
The file `tokenizer/spider_corpus.txt` is created on your hard drive. It looks like this:
```text
How many students are there?
SELECT count(*) FROM student
What is the budget of the IT department?
SELECT budget FROM department WHERE name = 'IT'
student
name
department
budget
...
```
This raw text file is what the BPE Tokenizer will read to learn exactly which words are common in our specific dataset!
