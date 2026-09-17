import torch
import os
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import math

from tokenizer.tokenizer import SpiderTokenizer
from embeddings.spider_dataset import SpiderDataset
from embeddings.collate import SpiderCollate
from transformer.transformer import Transformer
from training.loss import LabelSmoothingLoss


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """
    Creates a schedule with linear warmup followed by cosine decay.
    
    - For the first `num_warmup_steps`, the LR linearly ramps from 0 to the base LR.
    - After warmup, the LR follows a cosine curve down to 0.
    
    This is critical for Transformers because:
    1. Early in training, the attention weights are random garbage. 
       Large gradients at this stage would push them further into chaos.
    2. The warmup lets the model gently find a stable attention pattern first.
    3. The cosine decay then slowly reduces the LR so the model fine-tunes 
       its weights with smaller and smaller corrections.
    """
    def lr_lambda(current_step):
        # Linear warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)


def train():
    # 1. Hyperparameters
    batch_size = 8
    learning_rate = 5e-4          # Slightly higher base LR (warmup will handle the ramp)
    epochs = 30                   # Increased from 10 to 30 for proper convergence
    d_model = 512
    warmup_ratio = 0.1            # 10% of total steps used for warmup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    # Create output directory for checkpoints
    output_dir = "checkpoints"
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Load Data
    tokenizer = SpiderTokenizer("tokenizer/trained_tokenizer.json")
    
    dataset = SpiderDataset(
        data_path="data/spider_data/train_spider.json",
        tables_path="data/spider_data/tables.json",
        tokenizer=tokenizer
    )
    
    vocab_size = tokenizer.vocab_size
    padding_idx = tokenizer.pad_token_id
    
    collate_fn = SpiderCollate(pad_token_id=padding_idx)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    # 3. Initialize Model & Loss
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        src_pad_idx=padding_idx,
        tgt_pad_idx=padding_idx,
        d_model=d_model
    ).to(device)
    
    criterion = LabelSmoothingLoss(vocab_size=vocab_size, padding_idx=padding_idx).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # 4. Learning Rate Scheduler (Warmup + Cosine Decay)
    num_training_steps = len(dataloader) * epochs
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    
    print(f"Total training steps: {num_training_steps}")
    print(f"Warmup steps: {num_warmup_steps}")
    print(f"Saving checkpoints to: {output_dir}/")
    print("-" * 50)
    
    # 5. Training Loop
    model.train()
    global_step = 0
    best_loss = float("inf")
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            src = batch["encoder_input_ids"].to(device)
            tgt_input = batch["decoder_input_ids"].to(device)
            tgt_expected = batch["decoder_target_ids"].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(src=src, tgt=tgt_input)
            
            # Calculate Loss
            loss = criterion(logits, tgt_expected)
            
            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()  # Update learning rate every step
            
            total_loss += loss.item()
            global_step += 1
            
            if batch_idx % 50 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx} | Loss: {loss.item():.4f} | LR: {current_lr:.6f}")
                
        avg_loss = total_loss / len(dataloader)
        print(f"--- Epoch {epoch+1} Complete | Average Loss: {avg_loss:.4f} ---")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            path = os.path.join(output_dir, f"transformer_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), path)
            print(f"Checkpoint saved: {path}")
        
        # Always save the best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(output_dir, "transformer_best.pt")
            torch.save(model.state_dict(), best_path)
            print(f"New best model saved! (Loss: {best_loss:.4f})")
    
    # Save the final model
    final_path = os.path.join(output_dir, "transformer_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete! Final model saved: {final_path}")
    print(f"Best model loss: {best_loss:.4f}")

if __name__ == "__main__":
    train()
