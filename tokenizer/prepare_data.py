"""
Data preparation module for BPE tokenizer training.

Extracts natural language questions, SQL queries, and database schema metadata
from Spider dataset JSON files and writes a unified plain-text corpus file
(one sentence per line) for tokenizer training.

Usage:
    python -m tokenizer.prepare_data
    python -m tokenizer.prepare_data --data_dir data/spider_data --output data/tokenizer_corpus.txt
"""

import json
import argparse
from pathlib import Path
from typing import Optional


# Default paths relative to project root
DEFAULT_DATA_DIR = Path("data/spider_data")
DEFAULT_OUTPUT_PATH = Path("data/tokenizer_corpus.txt")


def load_json(filepath: Path) -> list[dict]:
    """Load a JSON file and return parsed data."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_nl_and_sql(examples: list[dict]) -> tuple[list[str], list[str]]:
    """
    Extract natural language questions and SQL queries from Spider examples.

    Args:
        examples: List of Spider example dicts with 'question' and 'query' keys.

    Returns:
        Tuple of (questions_list, queries_list).
    """
    questions = []
    queries = []

    for ex in examples:
        question = ex.get("question", "").strip()
        query = ex.get("query", "").strip()

        if question:
            questions.append(question)
        if query:
            queries.append(query)

    return questions, queries


def collect_db_ids(examples: list[dict]) -> set[str]:
    """
    Collect the set of database IDs referenced by a list of Spider examples.

    Args:
        examples: List of Spider example dicts with 'db_id' key.

    Returns:
        Set of database ID strings.
    """
    return {ex["db_id"] for ex in examples if ex.get("db_id")}


def extract_schema_metadata(
    tables: list[dict],
    db_id_filter: set[str] | None = None,
) -> list[str]:
    """
    Extract schema information (table names, column names) from tables.json.

    These schema tokens appear frequently in both NL questions and SQL queries,
    so including them ensures the tokenizer learns good subword units for them.

    Args:
        tables: List of database schema dicts from tables.json.
        db_id_filter: If provided, only include schemas for databases in this
                      set. Used to prevent data leakage by restricting to
                      training-only databases.

    Returns:
        List of schema description strings.
    """
    schema_lines = []

    for db in tables:
        db_id = db.get("db_id", "")

        # Skip databases not in the filter set (if a filter is provided)
        if db_id_filter is not None and db_id not in db_id_filter:
            continue

        table_names = db.get("table_names_original", [])
        column_names = db.get("column_names_original", [])

        # Add database identifier
        if db_id:
            schema_lines.append(db_id)

        # Add table names
        for table_name in table_names:
            if table_name:
                schema_lines.append(table_name)

        # Add column names (skip the wildcard '*' entry at index -1)
        for table_idx, col_name in column_names:
            if table_idx >= 0 and col_name and col_name != "*":
                schema_lines.append(col_name)

    return schema_lines


def prepare_corpus(
    data_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Prepare the text corpus from Spider dataset for tokenizer training.

    Uses only training data to avoid data leakage:
    - train_spider.json (7,000 examples)
    - train_others.json (1,659 examples)
    - tables.json — filtered to only include schemas for databases that
      appear in the training examples

    Dev/test data is deliberately excluded so the tokenizer's learned
    subword frequencies are not biased by evaluation data.

    Args:
        data_dir: Path to spider_data directory. Defaults to data/spider_data.
        output_path: Path to write the corpus file. Defaults to data/tokenizer_corpus.txt.

    Returns:
        Path to the written corpus file.
    """
    data_dir = data_dir or DEFAULT_DATA_DIR
    output_path = output_path or DEFAULT_OUTPUT_PATH

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Spider data directory not found at {data_dir}. "
            "Please download the dataset first."
        )

    corpus_lines: list[str] = []
    train_db_ids: set[str] = set()

    # --- 1. Extract from training data ---
    print("Loading train_spider.json...")
    train_spider = load_json(data_dir / "train_spider.json")
    questions, queries = extract_nl_and_sql(train_spider)
    train_db_ids |= collect_db_ids(train_spider)
    corpus_lines.extend(questions)
    corpus_lines.extend(queries)
    print(f"  Extracted {len(questions)} questions, {len(queries)} SQL queries")

    # --- 2. Extract from additional training data ---
    print("Loading train_others.json...")
    train_others = load_json(data_dir / "train_others.json")
    questions, queries = extract_nl_and_sql(train_others)
    train_db_ids |= collect_db_ids(train_others)
    corpus_lines.extend(questions)
    corpus_lines.extend(queries)
    print(f"  Extracted {len(questions)} questions, {len(queries)} SQL queries")

    print(f"\nTraining database IDs collected: {len(train_db_ids)}")

    # --- 3. Extract schema metadata (training DBs only) ---
    print("Loading tables.json (filtered to training DBs)...")
    tables = load_json(data_dir / "tables.json")
    schema_lines = extract_schema_metadata(tables, db_id_filter=train_db_ids)
    corpus_lines.extend(schema_lines)
    print(f"  Extracted {len(schema_lines)} schema entries "
          f"(from {len(train_db_ids)} training databases, "
          f"skipped {len(tables) - len(train_db_ids)} non-training databases)")

    # --- 5. Write corpus file ---
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for line in corpus_lines:
            # Clean up: strip whitespace, skip empty lines
            cleaned = line.strip()
            if cleaned:
                f.write(cleaned + "\n")

    total_lines = sum(1 for line in corpus_lines if line.strip())
    print(f"\nCorpus written to {output_path}")
    print(f"Total lines: {total_lines}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Spider dataset corpus for BPE tokenizer training."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Path to spider_data directory (default: data/spider_data)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output corpus file path (default: data/tokenizer_corpus.txt)",
    )
    args = parser.parse_args()

    prepare_corpus(data_dir=args.data_dir, output_path=args.output)


if __name__ == "__main__":
    main()
