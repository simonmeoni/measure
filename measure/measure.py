from rouge_score import scoring
from transformers import AutoModelForCausalLM, AutoTokenizer
from measure.text_attack import attack
from measure.fid_metric import Fid
import evaluate
import mauve
from rouge_score.rouge_scorer import RougeScorer
import sacrebleu


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


def semantic_metrics(predictions: list, references: list):

    # Use only 30% of predictions for MAUVE score calculation
    predictions_subset = predictions[: int(len(predictions) * 0.3)]

    mauve_score = mauve.compute_mauve(
        p_text=references,
        q_text=predictions_subset,
        device_id=0,
    )

    perplexity = evaluate.load("perplexity", module_type="metric")
    perplexity_results = perplexity.compute(
        predictions=predictions, model_id="gpt2", max_length=512
    )

    bertscore = evaluate.load("bertscore")
    bertscore_results = bertscore.compute(
        predictions=predictions, references=references, lang="en"
    )

    fid_calculator = Fid(model_name="all-mpnet-base-v2")
    fid_score = fid_calculator.compute(references, predictions)

    return {
        "mauve": mauve_score.mauve,
        "perplexity": perplexity_results["mean_perplexity"],
        "bertscore_f1": sum(bertscore_results["f1"]) / len(bertscore_results["f1"]),
        "fid": fid_score,
    }


def privacy_metrics(public_df, private_df, split=[80, 20, 20]):
    adaptive_results = attack(
        clean_df=public_df, private_df=private_df, split=split, attack_type="adaptive"
    )

    flattened_metrics = {}
    for field, metrics in adaptive_results.items():
        flattened_metrics[f"adaptive/f1"] = metrics["eval_F1 Score"]
        flattened_metrics[f"adaptive/mcc"] = metrics[
            "eval_Matthew correlation coefficient"
        ]

    static_results = attack(
        clean_df=public_df, private_df=private_df, split=split, attack_type="static"
    )
    for field, metrics in static_results.items():
        flattened_metrics[f"static/f1"] = metrics["eval_F1 Score"]
        flattened_metrics[f"static/mcc"] = metrics[
            "eval_Matthew correlation coefficient"
        ]

    return flattened_metrics
