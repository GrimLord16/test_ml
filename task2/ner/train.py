"""
train.py - Fine-tune DistilBERT for Animal Named Entity Recognition (NER)

Generates synthetic training data, fine-tunes a DistilBERT token classifier,
and saves the resulting model + tokenizer to disk.

Usage:
    python train.py --output_dir ./ner_model --num_epochs 5
"""

import argparse
import os
import sys

#In case it's run not from ner directory
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from datasets import DatasetDict

from data_utils import generate_ner_dataset, tokenize_and_align_labels, LABEL_NAMES


ID2LABEL = {i: name for i, name in enumerate(LABEL_NAMES)}
LABEL2ID = {name: i for i, name in ID2LABEL.items()}
NUM_LABELS = len(LABEL_NAMES)  # 5: O, B-ANIMAL, I-ANIMAL, B-NEG-ANIMAL, I-NEG-ANIMAL

def compute_metrics(eval_pred) -> dict:
    """Compute per-token accuracy and entity-level F1 (seqeval-style)."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)  # (batch, seq_len)

    true_labels, true_preds = [], []
    for pred_seq, label_seq in zip(predictions, labels):
        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id == -100:
                continue  # skip special / padding tokens
            true_labels.append(label_id)
            true_preds.append(pred_id)

    true_labels = np.array(true_labels)
    true_preds = np.array(true_preds)

    accuracy = (true_preds == true_labels).mean()

    def _prf(ids):
        tp = ((true_preds != 0) & (true_preds == true_labels) &
              np.isin(true_labels, list(ids))).sum()
        fp = (np.isin(true_preds, list(ids)) & (true_preds != true_labels)).sum()
        fn = (np.isin(true_labels, list(ids)) & (true_preds != true_labels)).sum()
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f = 2 * p * r / (p + r + 1e-9)
        return float(p), float(r), float(f)

    precision, recall, f1 = _prf({1, 2})
    neg_precision, neg_recall, neg_f1 = _prf({3, 4})

    return {
        "accuracy": float(accuracy),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "neg_precision": neg_precision,
        "neg_recall": neg_recall,
        "neg_f1": neg_f1,
    }


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT for Animal NER"
    )
    parser.add_argument(
        "--model_name",
        default="distilbert-base-uncased",
        help="HuggingFace model identifier (default: distilbert-base-uncased)",
    )
    parser.add_argument(
        "--output_dir",
        default="./ner_model",
        help="Directory to save the trained model (default: ./ner_model)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Training and evaluation batch size (default: 16)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="AdamW learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=2000,
        help="Number of synthetic training sentences to generate (default: 2000)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Animal NER — Training Configuration")
    print("=" * 60)
    for k, v in vars(args).items():
        print(f"  {k:20s}: {v}")
    print()

    # Generate synthetic NER dataset
    print(f"Generating {args.n_samples} synthetic NER sentences...")
    full_dataset = generate_ner_dataset(n_samples=args.n_samples, seed=42)
    print(f"Dataset created: {full_dataset}")

    split = full_dataset.train_test_split(test_size=0.2, seed=42)
    dataset_dict = DatasetDict({
        "train": split["train"],
        "validation": split["test"],
    })
    print(f"Train size: {len(dataset_dict['train'])}, "
          f"Validation size: {len(dataset_dict['validation'])}")

    print(f"\nLoading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    tokenized_dataset = dataset_dict.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer),
        batched=True,
        remove_columns=dataset_dict["train"].column_names,
    )
    print("Tokenization complete.")

    print(f"\nLoading model: {args.model_name} (num_labels={NUM_LABELS})")
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    import math
    steps_per_epoch = math.ceil(len(dataset_dict["train"]) / args.batch_size)
    warmup_steps = int(0.1 * steps_per_epoch * args.num_epochs)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",  
        fp16=False,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\nStarting training...")
    train_result = trainer.train()

    print("\nEvaluating on validation set...")
    eval_metrics = trainer.evaluate()

    print("\n" + "=" * 60)
    print("Training Metrics")
    print("=" * 60)
    for k, v in train_result.metrics.items():
        print(f"  {k:40s}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("Validation Metrics")
    print("=" * 60)
    for k, v in eval_metrics.items():
        print(f"  {k:40s}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save model and tokenizer
    print(f"\nSaving model and tokenizer to '{args.output_dir}'...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
