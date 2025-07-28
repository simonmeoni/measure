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


def _caml_classification(
    train_data_path: str = "data/caml/train_50.csv",
    dev_data_path: str = "data/caml/dev_50.csv",
    test_data_path: str = "data/caml/test_50.csv",
    model_name: str = "microsoft/deberta-v3-base",
    threshold: float = 0.5,
    seed: int = 42,
    **kwargs,
) -> Dict[str, float]:
    """
    CAML medical text classification using multi-label approach.

    Trains and evaluates a transformer model for medical code prediction
    on CAML discharge summaries.

    Args:
        train_data_path: Path to training CSV file
        dev_data_path: Path to development/validation CSV file
        test_data_path: Path to test CSV file
        model_name: Hugging Face model name for sequence classification
        threshold: Classification threshold for predictions
        seed: Random seed for reproducibility

    Returns:
        Dictionary of flattened metrics (accuracy, f1, precision, recall)
    """
    set_seed(seed)

    # Load datasets
    train_df = pd.read_csv(train_data_path)
    dev_df = pd.read_csv(dev_data_path)
    test_df = pd.read_csv(test_data_path)

    # Convert HOT_LABELS string representation to list
    def parse_hot_labels(hot_labels_str):
        # Remove brackets and split by comma, then convert to int
        labels_list = [float(x.strip()) for x in hot_labels_str.strip("[]").split(",")]
        return labels_list

    train_df["labels"] = train_df["HOT_LABELS"].apply(parse_hot_labels)
    dev_df["labels"] = dev_df["HOT_LABELS"].apply(parse_hot_labels)
    test_df["labels"] = test_df["HOT_LABELS"].apply(parse_hot_labels)

    # Rename TEXT column to text for consistency
    train_df["text"] = train_df["TEXT"]
    dev_df["text"] = dev_df["TEXT"]
    test_df["text"] = test_df["TEXT"]

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
    train_dataset = Dataset.from_pandas(train_df[["text", "labels"]]).map(
        tokenize_function, batched=True
    )
    dev_dataset = Dataset.from_pandas(dev_df[["text", "labels"]]).map(
        tokenize_function, batched=True
    )
    test_dataset = Dataset.from_pandas(test_df[["text", "labels"]]).map(
        tokenize_function, batched=True
    )

    # Convert labels to proper format
    def format_labels(example):
        example["labels"] = torch.Tensor(example["labels"])
        return example

    train_dataset = train_dataset.map(format_labels)
    dev_dataset = dev_dataset.map(format_labels)
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
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train model
    trainer.train()

    # Final evaluation on test set
    eval_results = trainer.evaluate(test_dataset)
    # Format results with consistent naming
    formatted_results = {}
    for key, value in eval_results.items():
        if key.startswith("eval_"):
            metric_name = key[5:]  # Remove 'eval_' prefix
            formatted_results[f"caml/{metric_name}"] = float(value)

    return formatted_results
