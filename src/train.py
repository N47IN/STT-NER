import os
import argparse
import torch
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from dataset import PIIDataset, collate_batch
from labels import LABELS
from model import create_model


def evaluate(model, dataloader, device):
    """Simple evaluation function."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = torch.tensor(batch["input_ids"], device=device)
            attention_mask = torch.tensor(batch["attention_mask"], device=device)
            labels = torch.tensor(batch["labels"], device=device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            
            # Calculate accuracy
            preds = torch.argmax(logits, dim=-1)
            mask = (labels != -100)
            correct = ((preds == labels) & mask).sum().item()
            total_correct += correct
            total_tokens += mask.sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / max(1, total_tokens)
    
    model.train()
    return avg_loss, accuracy


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="distilbert-base-uncased")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--dev", default="data/dev.jsonl")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("="*80)
    print("PII NER MODEL TRAINING")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print()

    # Load tokenizer and data
    print("Loading data...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = PIIDataset(args.train, tokenizer, LABELS, max_length=args.max_length, is_train=True)
    dev_ds = PIIDataset(args.dev, tokenizer, LABELS, max_length=args.max_length, is_train=False)
    
    print(f"Train examples: {len(train_ds)}")
    print(f"Dev examples: {len(dev_ds)}")
    print()

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_token_id=tokenizer.pad_token_id),
    )
    
    dev_dl = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, pad_token_id=tokenizer.pad_token_id),
    )

    # Create model
    print("Creating model...")
    model = create_model(args.model_name)
    model.to(args.device)
    model.train()

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    
    print(f"Total training steps: {total_steps}")
    print()
    
    # Training loop
    best_dev_loss = float('inf')
    history = []
    
    print("Starting training...")
    print("="*80)
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_ids = torch.tensor(batch["input_ids"], device=args.device)
            attention_mask = torch.tensor(batch["attention_mask"], device=args.device)
            labels = torch.tensor(batch["labels"], device=args.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_dl)
        
        # Evaluate on dev set
        dev_loss, dev_acc = evaluate(model, dev_dl, args.device)
        
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Dev Loss: {dev_loss:.4f}")
        print(f"  Dev Accuracy: {dev_acc:.4f}")
        
        # Save history
        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "dev_loss": dev_loss,
            "dev_accuracy": dev_acc
        })
        
        # Save best model
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            print(f"  ✅ Best model so far! Saving checkpoint...")
            model.save_pretrained(args.out_dir)
            tokenizer.save_pretrained(args.out_dir)
        
        # Save checkpoint for this epoch
        epoch_dir = os.path.join(args.out_dir, f"checkpoint-epoch-{epoch+1}")
        os.makedirs(epoch_dir, exist_ok=True)
        model.save_pretrained(epoch_dir)
        print(f"  Checkpoint saved: {epoch_dir}")
        print()

    # Save final model (last epoch)
    final_dir = os.path.join(args.out_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    # Save training history
    with open(os.path.join(args.out_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    
    print("="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Best model saved to: {args.out_dir}")
    print(f"Final model saved to: {final_dir}")
    print(f"Training history: {os.path.join(args.out_dir, 'training_history.json')}")
    print()


if __name__ == "__main__":
    main()
