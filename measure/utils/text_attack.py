from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn import metrics
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

Stats = Dict[str, Dict[str, float]]
TokenizerOutput = Dict[str, Any]

# Global configurations
TOKENIZER = AutoTokenizer.from_pretrained("bert-base-cased")
MAX_LENGTH = 512
STATIC_TRAINERS: Dict[str, Trainer] = {}


def split_df(split: List[int], df: pd.DataFrame) -> List[pd.DataFrame]:
    """Split DataFrame into train, validation, and test sets based on provided split ratios.
    Args:
        split: List of integers representing the split ratios for train, val, and test sets
        df: DataFrame to be split
    Returns:
        List of DataFrames for train, validation, and test sets
    """
    total_rows = len(df)
    random_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_size = int(total_rows * (split[0] / 100))
    val_size = int(total_rows * (split[1] / 100))

    return [
        random_df.iloc[:train_size].index.to_list(),
        random_df.iloc[train_size : train_size + val_size].index.to_list(),
        random_df.iloc[train_size + val_size :].index.to_list(),
    ]


def attack(
    clean_df: pd.DataFrame,
    private_df: pd.DataFrame,
    train_ds: pd.DataFrame,
    dev_ds: pd.DataFrame,
    test_ds: pd.DataFrame,
    attack_type: str,
    private_id_field: str = "private_id",
    clean_df_text_field: str = "text",
    private_df_text_field: str = "text",
) -> Stats:
    """Perform text attack analysis on datasets.

    Args:
        clean_df: DataFrame with clean text data
        private_df: DataFrame with private/perturbed text data
        clean_df_text_field: Column name for text in clean_df
        private_df_text_field: Column name for text in private_df
        train_idx_list: Indices for training set
        val_idx_list: Indices for validation set
        test_idx_list: Indices for test set
        attack_type: Type of attack ('static' or 'adaptive')
        private_id_field: Column name for private labels

    Returns:
        Dictionary containing attack statistics

    Raises:
        ValueError: If attack_type is not 'static' or 'adaptive'
        KeyError: If specified column fields don't exist in DataFrames
    """

    private_df_with_labels = pd.concat(
        [private_df, clean_df[[private_id_field]]], axis=1
    )

    # Initialize variables for type checking
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame

    # Use provided datasets instead of splitting
    train_idx_list = train_ds.index.tolist()
    val_idx_list = dev_ds.index.tolist()
    test_idx_list = test_ds.index.tolist()

    if attack_type == "static":
        train_df = clean_df[clean_df.index.isin(train_idx_list)]
        val_df = clean_df[clean_df.index.isin(val_idx_list)]
        test_df = private_df_with_labels[
            private_df_with_labels.index.isin(test_idx_list)
        ]

        train_text_field = clean_df_text_field
        val_text_field = clean_df_text_field
        test_text_field = private_df_text_field

    elif attack_type == "adaptive":
        train_df = private_df_with_labels[
            private_df_with_labels.index.isin(train_idx_list)
        ]
        val_df = private_df_with_labels[private_df_with_labels.index.isin(val_idx_list)]
        test_df = private_df_with_labels[
            private_df_with_labels.index.isin(test_idx_list)
        ]

        train_text_field = private_df_text_field
        val_text_field = private_df_text_field
        test_text_field = private_df_text_field
    else:
        raise ValueError(
            f"Invalid attack_type: {attack_type}. Must be 'static' or 'adaptive'"
        )

    train_df = train_df[[train_text_field, private_id_field]].dropna()
    val_df = val_df[[val_text_field, private_id_field]].dropna()
    test_df = test_df[[test_text_field, private_id_field]].dropna()

    stats = get_stats(
        train_df=train_df,
        test_df=test_df,
        val_df=val_df,
        train_text_field=train_text_field,
        val_text_field=val_text_field,
        test_text_field=test_text_field,
        private_id_field=private_id_field,
    )

    return stats


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Compute evaluation metrics for model predictions.

    Args:
        eval_pred: EvalPrediction object containing predictions and labels

    Returns:
        Dictionary containing accuracy, F1 score, and MCC
    """
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = np.argmax(logits, axis=-1)

    acc = metrics.accuracy_score(labels, predictions)
    f1 = metrics.f1_score(labels, predictions, average="macro")
    mcc = metrics.matthews_corrcoef(labels, predictions)

    stats = {"Accuracy": acc, "F1 Score": f1, "Matthew correlation coefficient": mcc}
    return stats


def get_trained_trainer(
    train_dataset: Dataset, val_dataset: Dataset, num_labels: int, label: str
) -> Trainer:
    """Get a trained BERT trainer for classification.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        num_labels: Number of classification labels
        label: Label identifier for caching

    Returns:
        Trained Trainer object
    """
    print(f"Number of labels: {num_labels}")
    if label in STATIC_TRAINERS:
        return STATIC_TRAINERS[label]

    training_args = TrainingArguments(
        output_dir=f"./output_{label}",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=30,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        logging_steps=300,
        load_best_model_at_end=True,
        metric_for_best_model="F1 Score",
    )

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=num_labels
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    STATIC_TRAINERS[label] = trainer
    return trainer


def run_eval(trainer: Trainer, test_dataset: Dataset) -> Dict[str, float]:
    """Run evaluation on test dataset.

    Args:
        trainer: Trained model trainer
        test_dataset: Test dataset

    Returns:
        Dictionary containing evaluation metrics
    """
    test_results = trainer.evaluate(eval_dataset=test_dataset)  # type: ignore
    return test_results


def get_stats(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_text_field: str,
    val_text_field: str,
    test_text_field: str,
    private_id_field: str = "private_id",
    verbose: bool = False,
) -> Stats:
    labels_fields = [private_id_field]
    label_field_stats: Dict[str, Dict[str, float]] = {}

    def tokenize_data(df: pd.DataFrame, text_field: str) -> TokenizerOutput:
        return TOKENIZER(
            df[text_field].tolist(),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )

    tokenized_train_data = tokenize_data(train_df, train_text_field)
    tokenized_val_data = tokenize_data(val_df, val_text_field)
    tokenized_test_data = tokenize_data(test_df, test_text_field)

    for label_field in labels_fields:
        num_classes: int = int(train_df[label_field].nunique())
        train_labels = torch.tensor(train_df[label_field].values, dtype=torch.long)
        val_labels = torch.tensor(val_df[label_field].values, dtype=torch.long)
        test_labels = torch.tensor(test_df[label_field].values, dtype=torch.long)

        train_dataset = Dataset.from_dict(
            {
                "input_ids": tokenized_train_data["input_ids"],
                "attention_mask": tokenized_train_data["attention_mask"],
                "labels": train_labels,
            }
        )

        val_dataset = Dataset.from_dict(
            {
                "input_ids": tokenized_val_data["input_ids"],
                "attention_mask": tokenized_val_data["attention_mask"],
                "labels": val_labels,
            }
        )

        test_dataset = Dataset.from_dict(
            {
                "input_ids": tokenized_test_data["input_ids"],
                "attention_mask": tokenized_test_data["attention_mask"],
                "labels": test_labels,
            }
        )

        trainer = get_trained_trainer(
            train_dataset, val_dataset, num_labels=num_classes, label=label_field
        )
        test_stats = run_eval(trainer, test_dataset)
        if verbose:
            print(f"Test statistics for {label_field}:")
            print(test_stats)
        label_field_stats[label_field] = test_stats

    return label_field_stats
