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

from measure.utils.gpt_eval import compute_preference, pairwise_mean, win_score_list_calculate


def gpt_eval(
    data_dir: str = "../data/alpacare",
    max_samples: int = 1000,
    tested_model_name="../data/model",
) -> Dict[str, float]:
    """
    Evaluate all models in the alpacare dataset against each other
    Following privacy.py philosophy

    Args:
        data_dir: Directory containing model outputs
        max_samples: Maximum number of samples to evaluate

    Returns:
        Dictionary of flattened metrics
    """

    # Fixed model names from alpacare subfolders
    model_names = ["claude-2", "gpt-3.5-turbo", "gpt-4", "text-davinci-003"]

    model_path = os.path.join(tested_model_name, "iCliniq_output.jsonl")
    all_metrics = {}

    # Compare each model against each reference
    for ref_name in model_names:
        if tested_model_name == ref_name:
            continue  # Skip self-comparison

        ref_path = os.path.join(data_dir, ref_name, "iCliniq_output.jsonl")

        print(f"Evaluating {tested_model_name} vs {ref_name}")

        try:
            # Get evaluations for both directions
            refer_first_data = compute_preference(
                model_data=model_path,
                reference_data=ref_path,
                model_name=tested_model_name,
                reference_name=ref_name,
                task_name="iCliniq",
                reference_first=True,
                batch_size=20,
                max_samples=max_samples,
                resume=True,
            )

            refer_last_data = compute_preference(
                model_data=model_path,
                reference_data=ref_path,
                model_name=tested_model_name,
                reference_name=ref_name,
                task_name="iCliniq",
                reference_first=False,
                batch_size=20,
                max_samples=max_samples,
                resume=True,
            )

            # Calculate win scores
            refer_first_score_arr = win_score_list_calculate(
                refer_first_data, tested_model_name, ref_name
            )
            refer_last_score_arr = win_score_list_calculate(
                refer_last_data, tested_model_name, ref_name
            )

            # Calculate final score
            score_mean = pairwise_mean(refer_first_score_arr, refer_last_score_arr)

            final_score = np.nanmean(score_mean)

            all_metrics[f"gpt_eval/{ref_name}"] = final_score
            all_metrics[f"gpt_eval/{ref_name}_refer_first"] = refer_first_data

        except Exception as e:
            print(f"Error evaluating {tested_model_name} vs {ref_name}: {e}")
            continue

    return all_metrics


def mimic_iii_icd_classification(
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


def caml_classification(
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
