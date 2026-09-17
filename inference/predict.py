import torch
from typing import List

from tokenizer.tokenizer import SpiderTokenizer
from transformer.transformer import Transformer

def greedy_search(model: torch.nn.Module, src: torch.Tensor, src_mask: torch.Tensor, bos_idx: int, eos_idx: int, max_len: int = 128) -> List[int]:
    """
    Fastest decoding. Picks the single most likely word at each step.
    """
    device = src.device
    
    # 1. Run the Encoder once
    enc_out = model.encode(src, src_mask)
    
    # 2. Start the decoder with just the [BOS] token
    tgt = torch.tensor([[bos_idx]], device=device)
    
    for _ in range(max_len):
        # 3. Create the causal mask for the current target sequence
        tgt_mask = model.make_tgt_mask(tgt)
        
        # 4. Run the Decoder
        out = model.decode(tgt, enc_out, tgt_mask, src_mask)
        
        # 5. Project to vocabulary logits
        logits = model.generator(out)
        
        # 6. Get the prediction for the last word (greedy: argmax)
        next_word_logits = logits[0, -1, :]
        next_word = next_word_logits.argmax().item()
        
        # 7. Append to our growing target sequence
        tgt = torch.cat([tgt, torch.tensor([[next_word]], device=device)], dim=1)
        
        # 8. Stop if the model generated [EOS]
        if next_word == eos_idx:
            break
            
    # Return the generated list of token IDs, excluding the initial [BOS]
    return tgt[0, 1:].tolist()

def generate_sql(question: str, schema: str, strategy: str = "greedy"):
    """
    Master function to chat with your AI!
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the Tokenizer
    tokenizer = SpiderTokenizer("tokenizer/trained_tokenizer.json")
    bos_idx = tokenizer.bos_token_id
    eos_idx = tokenizer.eos_token_id
    pad_idx = tokenizer.pad_token_id
    
    # 2. Initialize Model and load weights
    model = Transformer(
        src_vocab_size=tokenizer.vocab_size,
        tgt_vocab_size=tokenizer.vocab_size,
        src_pad_idx=pad_idx,
        tgt_pad_idx=pad_idx
    ).to(device)
    
    try:
        # Try loading the best model first, fall back to final
        model_path = "checkpoints/transformer_best.pt"
        if not torch.cuda.is_available():
            map_location = "cpu"
        else:
            map_location = None
            
        model.load_state_dict(torch.load(model_path, map_location=map_location))
        print(f"Loaded model weights from: {model_path}")
        model.eval()  # CRITICAL: Turn off Dropout!
    except FileNotFoundError:
        print("Warning: Model weights not found in checkpoints/. You need to run train.py first!")
        model.eval()
        
    # 3. Prepare the input exactly how the model expects it
    q_text = f"Question: {question}"
    s_text = f"Schema: {schema}"
    
    q_ids = tokenizer.encode(q_text)
    s_ids = tokenizer.encode(s_text)
    
    sep_idx = tokenizer._tokenizer.token_to_id("[SEP]")
    
    src_ids = [bos_idx] + q_ids + [sep_idx] + s_ids + [eos_idx]
    
    # Truncate if necessary (matching dataset logic)
    if len(src_ids) > 512: # Matching encoder_max_len
        # Keep question, truncate schema
        fixed_len = 3
        remaining = 512 - fixed_len - len(q_ids)
        if remaining > 0:
            s_ids = s_ids[:remaining]
            src_ids = [bos_idx] + q_ids + [sep_idx] + s_ids + [eos_idx]
        else:
            q_ids = q_ids[:512 - fixed_len]
            src_ids = [bos_idx] + q_ids + [eos_idx] # Drop schema entirely
        
    src = torch.tensor([src_ids], device=device)
    src_mask = model.make_src_mask(src)
    
    print(f"\nDecoding Strategy: {strategy.upper()}")
    
    # 4. Generate the SQL
    with torch.no_grad():  # Don't calculate gradients during inference!
        if strategy == "greedy":
            prediction_ids = greedy_search(model, src, src_mask, bos_idx, eos_idx)
        elif strategy == "beam":
            print("Beam Search not fully implemented yet, falling back to greedy!")
            prediction_ids = greedy_search(model, src, src_mask, bos_idx, eos_idx)
        else:
            raise ValueError("Strategy must be 'greedy' or 'beam'")
            
    # 5. Decode the IDs back into human-readable text
    predicted_sql = tokenizer.decode(prediction_ids, skip_special_tokens=True)
    
    print("-" * 50)
    print(f"Question: {question}")
    print(f"Generated SQL: \n{predicted_sql}")
    print("-" * 50)

if __name__ == "__main__":
    test_cases = [
        {
            "question": "How many singers do we have?",
            "schema": "singer (singer_id, name, country)"
        },
        {
            "question": "Show all names and countries of singers.",
            "schema": "singer (singer_id, name, country)"
        },
        {
            "question": "What is the average age of all students?",
            "schema": "student (stuid, lname, fname, age, sex, major, advisor, city_code)"
        },
        {
            "question": "Find the maximum capacity of all stadiums.",
            "schema": "stadium (stadium_id, location, name, capacity, highest, lowest, average)"
        }
    ]
    
    for test in test_cases:
        generate_sql(test["question"], test["schema"], strategy="greedy")
