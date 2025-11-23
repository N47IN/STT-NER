import os
import argparse
import torch
import json
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from dataset import PIIDataset, collate_batch
from labels import LABELS, ID2LABEL
from model import create_model


def evaluate(model, dataloader, device, tokenizer, id2label):
    """Evaluate using span-level metrics (proper for NER)."""
    from labels import label_is_pii
    
    model.eval()
    total_loss = 0.0
    
    # Collect all predictions and gold labels for span-level evaluation
    all_gold_spans = []
    all_pred_spans = []
    
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
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            
            # Convert to spans for each example in batch
            for i, (pred_seq, label_seq, text, offsets) in enumerate(zip(
                preds, batch["labels"], batch["texts"], batch["offset_mapping"]
            )):
                # Convert BIO predictions to spans
                pred_spans = bio_to_spans(text, offsets, pred_seq, id2label)
                # Convert gold labels to spans
                gold_spans = bio_to_spans(text, offsets, label_seq, id2label)
                
                all_pred_spans.append(pred_spans)
                all_gold_spans.append(gold_spans)
    
    avg_loss = total_loss / len(dataloader)
    
    # Compute span-level metrics
    tp_pii = fp_pii = fn_pii = 0
    tp_non = fp_non = fn_non = 0
    
    for pred_spans, gold_spans in zip(all_pred_spans, all_gold_spans):
        gold_pii = set((s, e, "PII") for s, e, lab in gold_spans if label_is_pii(lab))
        gold_non = set((s, e, "NON") for s, e, lab in gold_spans if not label_is_pii(lab))
        pred_pii = set((s, e, "PII") for s, e, lab in pred_spans if label_is_pii(lab))
        pred_non = set((s, e, "NON") for s, e, lab in pred_spans if not label_is_pii(lab))
        
        # PII metrics
        for span in pred_pii:
            if span in gold_pii:
                tp_pii += 1
            else:
                fp_pii += 1
        for span in gold_pii:
            if span not in pred_pii:
                fn_pii += 1
        
        # Non-PII metrics
        for span in pred_non:
            if span in gold_non:
                tp_non += 1
            else:
                fp_non += 1
        for span in gold_non:
            if span not in pred_non:
                fn_non += 1
    
    # Calculate PII precision (most important metric)
    pii_precision = tp_pii / max(1, tp_pii + fp_pii)
    pii_recall = tp_pii / max(1, tp_pii + fn_pii)
    pii_f1 = 2 * pii_precision * pii_recall / max(1e-10, pii_precision + pii_recall)
    
    model.train()
    return {
        "loss": avg_loss,
        "pii_precision": pii_precision,
        "pii_recall": pii_recall,
        "pii_f1": pii_f1
    }


def bio_to_spans(text, offsets, label_ids, id2label):
    """Convert BIO label sequence to entity spans."""
    spans = []
    current_label = None
    current_start = None
    current_end = None
    
    for (start, end), lid in zip(offsets, label_ids):
        if start == 0 and end == 0:
            continue  # Special token
        label = id2label.get(int(lid), "O")
        if label == "O":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
                current_label = None
            continue
        
        prefix, ent_type = label.split("-", 1)
        if prefix == "B":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
            current_label = ent_type
            current_start = start
            current_end = end
        elif prefix == "I":
            if current_label == ent_type:
                current_end = end
            else:
                if current_label is not None:
                    spans.append((current_start, current_end, current_label))
                current_label = ent_type
                current_start = start
                current_end = end
    
    if current_label is not None:
        spans.append((current_start, current_end, current_label))
    
    return spans


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
    ap.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing factor (reduces overconfidence)")
    ap.add_argument("--classifier_dropout", type=float, default=0.2, help="Dropout probability for classifier")
    ap.add_argument("--early_stopping_patience", type=int, default=3, help="Stop if no improvement for N epochs")
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

    # Create model with enhanced regularization
    print("Creating model with dropout regularization...")
    model = create_model(args.model_name, classifier_dropout=args.classifier_dropout)
    model.to(args.device)
    model.train()
    
    print(f"  Classifier dropout: {args.classifier_dropout}")
    print(f"  Label smoothing: {args.label_smoothing}")
    print(f"  Early stopping patience: {args.early_stopping_patience}")
    
    # Create loss function with label smoothing
    loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index=-100,
        label_smoothing=args.label_smoothing
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    
    print(f"Total training steps: {total_steps}")
    print()
    
    # Training loop with early stopping
    best_dev_precision = 0.0  # Track best PII precision
    epochs_without_improvement = 0
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

            # Forward pass (without computing loss internally)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Compute loss with label smoothing
            # Reshape for loss computation: (batch * seq_len, num_labels) vs (batch * seq_len)
            loss = loss_fn(logits.view(-1, len(LABELS)), labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_dl)
        
        # Evaluate on dev set (span-level metrics)
        dev_metrics = evaluate(model, dev_dl, args.device, tokenizer, ID2LABEL)
        
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Dev Loss: {dev_metrics['loss']:.4f}")
        print(f"  PII Precision: {dev_metrics['pii_precision']:.4f} (target: ≥0.80)")
        print(f"  PII Recall: {dev_metrics['pii_recall']:.4f}")
        print(f"  PII F1: {dev_metrics['pii_f1']:.4f}")
        
        # Save history
        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "dev_loss": dev_metrics['loss'],
            "pii_precision": dev_metrics['pii_precision'],
            "pii_recall": dev_metrics['pii_recall'],
            "pii_f1": dev_metrics['pii_f1']
        })
        
        # Save best model based on PII precision (most important metric)
        current_pii_precision = dev_metrics['pii_precision']
        if current_pii_precision > best_dev_precision:
            best_dev_precision = current_pii_precision
            epochs_without_improvement = 0
            print(f"  ✅ Best PII precision so far! Saving checkpoint...")
            model.save_pretrained(args.out_dir)
            tokenizer.save_pretrained(args.out_dir)
        else:
            epochs_without_improvement += 1
            print(f"  ⚠️  No improvement for {epochs_without_improvement} epoch(s)")
        
        # Save checkpoint for this epoch
        epoch_dir = os.path.join(args.out_dir, f"checkpoint-epoch-{epoch+1}")
        os.makedirs(epoch_dir, exist_ok=True)
        model.save_pretrained(epoch_dir)
        print(f"  Checkpoint saved: {epoch_dir}")
        
        # Early stopping check
        if epochs_without_improvement >= args.early_stopping_patience:
            print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
            print(f"  No improvement in PII precision for {args.early_stopping_patience} epochs")
            print(f"  Best PII precision: {best_dev_precision:.4f}")
            break
        
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
