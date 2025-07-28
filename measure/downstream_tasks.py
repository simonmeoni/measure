from typing import Dict

from measure.utils.downstream_tasks.caml import _caml_classification
from measure.utils.downstream_tasks.gpt_eval import _gpt_eval
from measure.utils.downstream_tasks.mimic_iii import _mimic_iii_icd_classification


def gpt_eval(
    data_dir: str = "../data/alpacare",
    max_samples: int = 1000,
    tested_model_name="../data/model",
) -> Dict[str, float]:
    return _gpt_eval(data_dir, max_samples, tested_model_name)


def mimic_iii_icd_classification(
    test_data_dir: str = "data/mimic-iii-test",
    train_data_dir: str = "data/mimic-iii-train",
    model_name: str = "microsoft/deberta-v3-base",
    threshold: float = 0.5,
    seed: int = 42,
    **kwargs,
) -> Dict[str, float]:
    return _mimic_iii_icd_classification(
        test_data_dir, train_data_dir, model_name, threshold, seed, **kwargs
    )


def caml_classification(
    train_data_path: str = "data/caml/train_50.csv",
    dev_data_path: str = "data/caml/dev_50.csv",
    test_data_path: str = "data/caml/test_50.csv",
    model_name: str = "microsoft/deberta-v3-base",
    threshold: float = 0.5,
    seed: int = 42,
    **kwargs,
) -> Dict[str, float]:
    return _caml_classification(
        train_data_path,
        dev_data_path,
        test_data_path,
        model_name,
        threshold,
        seed,
        **kwargs,
    )
