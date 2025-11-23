"""
Improved training script with:
- Validation loop
- Early stopping
- Best model checkpoint
- Class weighting
- Gradient clipping
- Comprehensive logging
"""

import os
import argparse
import torch
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from collections import Counter
import numpy as np

from dataset import PIIDataset, collate_batch
from labels import LABELS, label_is_pii
from model import create_model


def compute_class_weights(train_dataset, num_labels):
    """Compute class weights for imbalanced dataset."""
    label_counts = Counter()
    for item in train_dataset.items:
        for label_id in item['labels']:
            if label_id != -100:  # Ignore padding
                label_counts[label_id] += 1
    
    total = sum(label_counts.values())
    weights = []
    for i in range(num_labels):
        count = label_counts.get(i, 1)
        # Inverse frequency weighting
        weight = total / (count * num_labels)
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32)


def evaluate_model(model, dataloader, device, label_list):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = torch.tensor(batch["input_ids"], device=device)
            attention_mask = torch.tensor(batch["attention_mask"], device=device)
            labels = torch.tensor(batch["labels"], device=device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            
            # Get predictions
            preds = torch.argmax(logits, dim=-1)
            
            # Collect predictions and labels (exclude padding)
            for pred_seq, label_seq, mask in zip(preds, labels, attention_mask):
                for p, l, m in zip(pred_seq, label_seq, mask):
                    if m == 1 and l != -100:
                        all_preds.append(p.item())
                        all_labels.append(l.item())
    
    avg_loss = total_loss / max(1, len(dataloader))
    
    # Compute accuracy
    correct = sum([1 for p, l in zip(all_preds, all_labels) if p == l])
    accuracy = correct / max(1, len(all_preds))
    
    # Compute per-class metrics
    label2id = {label: i for i, label in enumerate(label_list)}
    pii_labels = set([label2id[f"B-{entity}"] for entity in ["CREDIT_CARD", "PHONE", "EMAIL", "PERSON_NAME", "DATE"] if f"B-{entity}" in label2id])
    pii_labels.update([label2id[f"I-{entity}"] for entity in ["CREDIT_CARD", "PHONE", "EMAIL", "PERSON_NAME", "DATE"] if f"I-{entity}" in label2id])
    
    pii_correct = sum([1 for p, l in zip(all_preds, all_labels) if p == l and l in pii_labels])
    pii_total = sum([1 for l in all_labels if l in pii_labels])
    pii_pred_total = sum([1 for p in all_preds if p in pii_labels])
    
    pii_precision = pii_correct / max(1, pii_pred_total)
    pii_recall = pii_correct / max(1, pii_total)
    pii_f1 = 2 * pii_precision * pii_recall / max(1e-10, pii_precision + pii_recall)
    
    model.train()
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "pii_precision": pii_precision,
        "pii_recall": pii_recall,
        "pii_f1": pii_f1
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="distilbert-base-uncased")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--dev", default="data/dev.jsonl")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--use_class_weights", action="store_true", default=True)
    ap.add_argument("--early_stopping_patience", type=int, default=3)
    ap.add_argument("--gradient_clip", type=float, default=1.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("="*80)
    print("PII NER TRAINING - IMPROVED PIPELINE")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_name}")
    print(f"  Device: {args.device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Max length: {args.max_length}")
    print(f"  Early stopping patience: {args.early_stopping_patience}")
    print(f"  Class weights: {args.use_class_weights}")
    print(f"  Gradient clipping: {args.gradient_clip}")
    print()
    
    # Load tokenizer and datasets
    print("Loading data...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    train_ds = PIIDataset(args.train, tokenizer, LABELS, max_length=args.max_length, is_train=True)
    dev_ds = PIIDataset(args.dev, tokenizer, LABELS, max_length=args.max_length, is_train=False)
    
    print(f"  Train examples: {len(train_ds)}")
    print(f"  Dev examples: {len(dev_ds)}")
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
    
    # Compute class weights
    if args.use_class_weights:
        print("Computing class weights...")
        class_weights = compute_class_weights(train_ds, len(LABELS))
        class_weights = class_weights.to(args.device)
        print(f"  Class weights computed (O weight: {class_weights[0]:.3f}, entity avg: {class_weights[1:].mean():.3f})")
    else:
        class_weights = None
    
    # Override model's loss function with class weights
    if class_weights is not None:
        def compute_loss(logits, labels):
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
            return loss_fct(logits.view(-1, len(LABELS)), labels.view(-1))
        
        # Monkey patch
        original_forward = model.forward
        def forward_with_weighted_loss(*args, **kwargs):
            outputs = original_forward(*args, **kwargs)
            if 'labels' in kwargs and kwargs['labels'] is not None:
                outputs.loss = compute_loss(outputs.logits, kwargs['labels'])
            return outputs
        model.forward = forward_with_weighted_loss
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    total_steps = len(train_dl) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"\nTraining setup:")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print()
    
    # Training loop with early stopping
    best_pii_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    training_history = []
    
    print("Starting training...")
    print("="*80)
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in progress_bar:
            input_ids = torch.tensor(batch["input_ids"], device=args.device)
            attention_mask = torch.tensor(batch["attention_mask"], device=args.device)
            labels = torch.tensor(batch["labels"], device=args.device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_train_loss = running_loss / len(train_dl)
        
        # Evaluate on dev set
        print(f"\nEvaluating epoch {epoch+1}...")
        dev_metrics = evaluate_model(model, dev_dl, args.device, LABELS)
        
        print(f"Epoch {epoch+1} Results:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Dev Loss: {dev_metrics['loss']:.4f}")
        print(f"  Dev Accuracy: {dev_metrics['accuracy']:.4f}")
        print(f"  PII Precision: {dev_metrics['pii_precision']:.4f}")
        print(f"  PII Recall: {dev_metrics['pii_recall']:.4f}")
        print(f"  PII F1: {dev_metrics['pii_f1']:.4f}")
        
        # Save history
        training_history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "dev_loss": dev_metrics['loss'],
            "dev_accuracy": dev_metrics['accuracy'],
            "pii_precision": dev_metrics['pii_precision'],
            "pii_recall": dev_metrics['pii_recall'],
            "pii_f1": dev_metrics['pii_f1']
        })
        
        # Check for improvement
        if dev_metrics['pii_f1'] > best_pii_f1:
            best_pii_f1 = dev_metrics['pii_f1']
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save best model
            print(f"  ✅ New best model! (PII F1: {best_pii_f1:.4f})")
            model.save_pretrained(args.out_dir)
            tokenizer.save_pretrained(args.out_dir)
            
            # Save metrics
            with open(os.path.join(args.out_dir, "best_metrics.json"), "w") as f:
                json.dump(dev_metrics, f, indent=2)
        else:
            patience_counter += 1
            print(f"  No improvement (patience: {patience_counter}/{args.early_stopping_patience})")
            
            if patience_counter >= args.early_stopping_patience:
                print(f"\n⏹ Early stopping triggered at epoch {epoch+1}")
                break
        
        print()
    
    # Save training history
    with open(os.path.join(args.out_dir, "training_history.json"), "w") as f:
        json.dump(training_history, f, indent=2)
    
    print("="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nBest model from epoch {best_epoch}")
    print(f"Best PII F1: {best_pii_f1:.4f}")
    print(f"\nModel saved to: {args.out_dir}")
    print(f"Training history saved to: {os.path.join(args.out_dir, 'training_history.json')}")
    print()


if __name__ == "__main__":
    main()

