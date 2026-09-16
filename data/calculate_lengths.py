import json
from pathlib import Path
import sys
import numpy as np
sys.path.append('/home/prasanna/Documents/my_projects/encoder_decoder_transformer')
from tokenizer import SpiderTokenizer

def get_schema_string(db_id, tables):
    # Find the db
    db = next((d for d in tables if d['db_id'] == db_id), None)
    if not db:
        return ""
    
    table_names = db['table_names_original']
    column_names = db['column_names_original']
    
    # group columns by table
    # column_names is [[table_idx, col_name], ...]
    # table_idx -1 is *
    table_to_cols = {i: [] for i in range(len(table_names))}
    for t_idx, col_name in column_names:
        if t_idx >= 0:
            table_to_cols[t_idx].append(col_name)
            
    schema_parts = []
    for t_idx, t_name in enumerate(table_names):
        cols = table_to_cols[t_idx]
        col_str = ", ".join(cols)
        schema_parts.append(f"{t_name} ({col_str})")
        
    return " | ".join(schema_parts)

def main():
    tok = SpiderTokenizer("tokenizer/trained_tokenizer.json")
    
    with open("data/spider_data/tables.json") as f:
        tables = json.load(f)
        
    with open("data/spider_data/train_spider.json") as f:
        train_data = json.load(f)
        
    with open("data/spider_data/train_others.json") as f:
        train_data.extend(json.load(f))
        
    encoder_lengths = []
    decoder_lengths = []
    
    for ex in train_data:
        question = ex['question']
        query = ex['query']
        db_id = ex['db_id']
        
        schema_str = get_schema_string(db_id, tables)
        
        # Format: Question: <question> [SEP] Schema: <schema>
        # Let's just tokenize question and schema separately and add 3 for BOS, SEP, EOS
        q_ids = tok.encode("Question: " + question)
        s_ids = tok.encode("Schema: " + schema_str)
        
        # length = BOS + q_ids + SEP + s_ids + EOS
        enc_len = 1 + len(q_ids) + 1 + len(s_ids) + 1
        encoder_lengths.append(enc_len)
        
        # decoder = BOS + query (target is query + EOS, same length)
        dec_len = 1 + len(tok.encode(query))
        decoder_lengths.append(dec_len)
        
    print(f"Total examples: {len(encoder_lengths)}")
    print("\nEncoder Lengths (Question + Schema):")
    print(f"Min: {np.min(encoder_lengths)}")
    print(f"Mean: {np.mean(encoder_lengths):.1f}")
    print(f"Median: {np.median(encoder_lengths)}")
    print(f"95th: {np.percentile(encoder_lengths, 95)}")
    print(f"99th: {np.percentile(encoder_lengths, 99)}")
    print(f"Max: {np.max(encoder_lengths)}")
    
    print("\nDecoder Lengths (SQL Query):")
    print(f"Min: {np.min(decoder_lengths)}")
    print(f"Mean: {np.mean(decoder_lengths):.1f}")
    print(f"Median: {np.median(decoder_lengths)}")
    print(f"95th: {np.percentile(decoder_lengths, 95)}")
    print(f"99th: {np.percentile(decoder_lengths, 99)}")
    print(f"Max: {np.max(decoder_lengths)}")

if __name__ == '__main__':
    main()
