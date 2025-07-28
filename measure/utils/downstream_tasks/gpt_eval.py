import os
from typing import Dict

import numpy as np

from measure.utils.gpt_eval import compute_preference, pairwise_mean, win_score_list_calculate


def _gpt_eval(
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
