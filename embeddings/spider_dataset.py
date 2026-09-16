

import json
from typing import Dict, List
import torch
from torch.utils.data import Dataset

from tokenizer import SpiderTokenizer

class SpiderDataset(Dataset):
    """
    PyTorch Dataset for Spider text-to-SQL.
    
    Responsible for:
    1. Loading and formatting database schemas and questions.
    2. Explicitly managing special tokens ([BOS], [SEP], [EOS]).
    3. Applying robust sequence truncation that preserves critical tokens.
    4. Providing properly formatted encoder and decoder sequences.
    """

    def __init__(
        self,
        data_path: str,
        tables_path: str,
        tokenizer: SpiderTokenizer,
        encoder_max_len: int = 512,     #  this is the max size of context window  of  encoder  ( since no query will be more than 512 tokens)
        decoder_max_len: int = 128,):   #  this is the max size of context window  of  decoder  ( since no query will be more than 128 tokens)

        self.tokenizer = tokenizer      # the tokenizer that we are going to use 
        self.encoder_max_len = encoder_max_len
        self.decoder_max_len = decoder_max_len

        # Load raw data
        with open(data_path, "r", encoding="utf-8") as f:
            self.examples = json.load(f)

        with open(tables_path, "r", encoding="utf-8") as f:
            tables_raw = json.load(f)

        # Precompute schema strings for fast lookup
        self.schemas: Dict[str, str] = self._build_schemas(tables_raw)
        

    def _build_schemas(self, tables_raw: List[Dict]) -> Dict[str, str]:
        """
        Builds a deterministic schema string for each database.
        Format: table1 (col1, col2) | table2 (col1)
        Preserves the original order from Spider's tables.json.
        """
        schemas = {}
        for db in tables_raw:
            db_id = db["db_id"]
            table_names = db["table_names_original"]
            column_names = db["column_names_original"]

            # Group columns by table index (idx -1 is *)
            table_to_cols = {i: [] for i in range(len(table_names))}
            for t_idx, col_name in column_names:
                if t_idx >= 0:
                    table_to_cols[t_idx].append(col_name)

            # Build string
            schema_parts = []
            for t_idx, t_name in enumerate(table_names):
                cols = table_to_cols[t_idx]
                col_str = ", ".join(cols)
                schema_parts.append(f"{t_name} ({col_str})")

            schemas[db_id] = " | ".join(schema_parts)
            
        return schemas

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        example = self.examples[idx]
        
        # 1. Extract raw text
        question = example["question"]
        query = example["query"]
        db_id = example["db_id"]
        
        if db_id not in self.schemas:
            raise KeyError(f"Database ID '{db_id}' found in examples but missing from tables.json")
        schema = self.schemas[db_id]

        # 2. Add prefixes for context
        q_text = f"Question: {question}"
        s_text = f"Schema: {schema}"

        # 3. Tokenize independently (no special tokens yet)
        q_ids = self.tokenizer.encode(q_text)
        s_ids = self.tokenizer.encode(s_text)
        sql_ids = self.tokenizer.encode(query)

        # 4. Fetch special token IDs
        bos = self.tokenizer.bos_token_id
        eos = self.tokenizer.eos_token_id
        sep = self.tokenizer.sep_token_id

        # ---------------------------------------------------------
        # ENCODER FORMATTING: [BOS] Question [SEP] Schema [EOS]
        # ---------------------------------------------------------
        # Truncation logic: We prioritize preserving the full question. 
        # If the combined sequence exceeds encoder_max_len, we truncate 
        # the schema first. If the question itself is too long (rare edge case), 
        # it is truncated as a last resort. 
        # [BOS], [SEP], and [EOS] are always preserved.
        
        fixed_len = 3 # BOS, SEP, EOS
        
        if fixed_len + len(q_ids) + len(s_ids) > self.encoder_max_len:
            # How much space is left for the schema?
            remaining = self.encoder_max_len - fixed_len - len(q_ids)
            
            if remaining > 0:
                s_ids = s_ids[:remaining]
            else:
                # Extreme edge case: question itself is too long.
                # Drop the schema entirely and truncate the question.
                s_ids = []
                q_ids = q_ids[:self.encoder_max_len - fixed_len]

        encoder_input = [bos] + q_ids + [sep] + s_ids + [eos]

        # ---------------------------------------------------------
        # DECODER FORMATTING (Teacher Forcing)
        # Input:  [BOS] SQL
        # Target: SQL [EOS]
        # ---------------------------------------------------------
        # Both input and target use max 1 special token, so we check sql_ids + 1
        
        if len(sql_ids) + 1 > self.decoder_max_len:
            sql_ids = sql_ids[:self.decoder_max_len - 1]

        decoder_input = [bos] + sql_ids
        decoder_target = sql_ids + [eos]

        # Sanity checks (ensure we never violate max lengths)
        assert len(encoder_input) <= self.encoder_max_len
        assert len(decoder_input) <= self.decoder_max_len
        assert len(decoder_target) <= self.decoder_max_len
        
        # Ensure target lengths match input lengths exactly
        assert len(decoder_input) == len(decoder_target)

        return {
            "encoder_input_ids": encoder_input,
            "decoder_input_ids": decoder_input,
            "decoder_target_ids": decoder_target,
        }
