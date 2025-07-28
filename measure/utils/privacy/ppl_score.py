from typing import List

import evaluate
import numpy as np


def _compute_ppl_scores(
    texts: List[str],
    model_name_or_path: str,
    batch_size: int = 8,
) -> np.ndarray:
    ppl = evaluate.load("perplexity", module_type="metric")
    # compute retourne un dict avec clé "perplexities"
    vals = ppl.compute(predictions=texts, model_id="gpt2", max_length=512)
    print(vals)
    return np.array(vals["perplexities"], dtype=float)
