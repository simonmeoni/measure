import os
from typing import Dict

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


def _mimic_iii_icd_classification(
    test_data_dir: str = "data/mimic-iii-test",
    train_data_dir: str = "data/mimic-iii-train",
    model_name: str = "microsoft/deberta-v3-base",
    threshold: float = 0.5,
    seed: int = 42,
    **kwargs,
) -> Dict[str, float]:
    """
    MIMIC-III medical text classification using multi-label approach.

    Trains and evaluates a transformer model for ICD-9 code prediction
    on MIMIC-III discharge summaries.

    Args:
        data_dir: Directory containing train.parquet and test.parquet
        model_name: Hugging Face model name for sequence classification
        threshold: Classification threshold for predictions
        seed: Random seed for reproducibility

    Returns:
        Dictionary of flattened metrics (accuracy, f1, precision, recall)
    """
    set_seed(seed)

    # Load datasets
    train_path = os.path.join(train_data_dir, "train.parquet")
    test_path = os.path.join(test_data_dir, "test.parquet")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # Get number of labels from training data
    num_labels = len(train_df.iloc[0]["labels"])

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, problem_type="multi_label_classification"
    )

    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, padding=True, max_length=512
        )

    # Convert to HF datasets and tokenize
    train_dataset = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
    test_dataset = Dataset.from_pandas(test_df).map(tokenize_function, batched=True)

    # Convert labels to proper format
    def format_labels(example):
        example["labels"] = torch.Tensor(example["labels"])
        return example

    train_dataset = train_dataset.map(format_labels)
    test_dataset = test_dataset.map(format_labels)

    # Setup metrics
    metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = sigmoid(predictions)
        predictions = predictions > threshold
        labels = labels.astype(float)

        result = metrics.compute(
            predictions=predictions.astype(float).reshape(-1),
            references=labels.reshape(-1),
        )
        return result

    # Training arguments using provided configuration
    training_args = TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=20,
        warmup_steps=50,
        eval_steps=100,
        evaluation_strategy="steps",
        remove_unused_columns=True,
        save_strategy="no",
        output_dir="./tmp_output",
        num_train_epochs=10,
        learning_rate=2e-5,
        report_to="none",
    )

    # Setup trainer
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train model
    trainer.train()

    # Final evaluation
    eval_results = trainer.evaluate()

    # Format results with consistent naming
    formatted_results = {}
    for key, value in eval_results.items():
        if key.startswith("eval_"):
            metric_name = key[5:]  # Remove 'eval_' prefix
            formatted_results[f"mimic_iii_icd/{metric_name}"] = float(value)

    return formatted_results
