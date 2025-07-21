import os
import re
from typing import Dict

import numpy as np

from measure.utils.gpt_eval import gpt_evaluate


def remove_special_characters(input_string):
    """Remove special characters from string"""
    cleaned_string = re.sub(r"[^a-zA-Z0-9]", "", input_string)
    return cleaned_string


def str_win_calculate(completion, compared_model, refer_model):
    """Calculate win score from string completion"""
    if completion == refer_model:
        return [0]
    elif completion == compared_model:
        return [1]
    elif "Tie" in completion:
        return [0.5]
    else:
        return [np.nan]


def win_score_list_calculate(data, compared_model, refer_model):
    """Calculate win scores from evaluation data"""
    score_list = []
    for d_i in data:
        output = d_i["response"]
        model_1 = d_i["model_1"]
        model_2 = d_i["model_2"]

        output = output.strip()
        output = output.replace("Output (a)", model_1)
        output = output.replace("Output (b)", model_2)
        compared_model = remove_special_characters(compared_model)
        refer_model = remove_special_characters(refer_model)
        output = remove_special_characters(output)

        score = str_win_calculate(output, compared_model, refer_model)
        score_list.extend(score)

    return np.array(score_list)


def pairwise_mean(arr1, arr2):
    """Calculate pairwise mean of two arrays"""
    assert len(arr1) == len(arr2), "Arrays must be of the same length"
    means = np.where((~np.isnan(arr1) & ~np.isnan(arr2)), (arr1 + arr2) / 2, np.nan)
    return means


def gpt_eval_alpacare(
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
            refer_first_data = gpt_evaluate(
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

            refer_last_data = gpt_evaluate(
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
