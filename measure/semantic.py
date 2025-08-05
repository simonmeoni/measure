import evaluate
import mauve
import sacrebleu
from rouge_score import scoring
from rouge_score.rouge_scorer import RougeScorer

from measure.utils.semantic.fid import _Fid


def translation_metrics(predictions: list, references: list):
    rouge_score = RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()
    for pred, ref in zip(predictions, references):
        result = rouge_score.score(ref, pred)
        aggregator.add_scores(result)
    rouge_scores = aggregator.aggregate()
    bleu_score = sacrebleu.corpus_bleu(predictions, [references])

    return {
        "rouge1": rouge_scores["rouge1"].mid.fmeasure,
        "rouge2": rouge_scores["rouge2"].mid.fmeasure,
        "rougeL": rouge_scores["rougeL"].mid.fmeasure,
        "bleu": round(bleu_score.score, ndigits=3),
    }


def similarity_metrics(
    predictions: list, references: list, mauve_scaling_factor: int = 5
):

    model_id = "sentence-transformers/paraphrase-MiniLM-L6-v2"
    mauve_score = mauve.compute_mauve(
        p_text=references,
        q_text=predictions,
        num_buckets=int(0.1 * len(predictions)),
        mauve_scaling_factor=mauve_scaling_factor,
        device_id=0,
        featurize_model_name=model_id,
        verbose=False,
        max_text_length=1024,
    )

    perplexity = evaluate.load("perplexity", module_type="metric")
    predictions_perplexity_results = perplexity.compute(
        predictions=predictions, model_id="gpt2", max_length=1024
    )
    references_perplexity_results = perplexity.compute(
        predictions=references, model_id="gpt2", max_length=1024
    )

    fid_calculator = _Fid(model_name=model_id)
    fid_score = fid_calculator.compute(references, predictions)

    return {
        "mauve": mauve_score.mauve,
        "references_perplexity": references_perplexity_results["mean_perplexity"],
        "predictions_perplexity": predictions_perplexity_results["mean_perplexity"],
        "fid": fid_score,
    }
