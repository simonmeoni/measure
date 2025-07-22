from pycanon import anonymity as an

from measure.utils.text_attack import attack


def author_attack(public_df, private_df, split=[80, 20, 20]):
    adaptive_results = attack(
        clean_df=public_df, private_df=private_df, split=split, attack_type="adaptive"
    )

    flattened_metrics = {}
    for _, metrics in adaptive_results.items():
        flattened_metrics["adaptive/f1"] = metrics["eval_F1 Score"]
        flattened_metrics["adaptive/mcc"] = metrics[
            "eval_Matthew correlation coefficient"
        ]

    static_results = attack(
        clean_df=public_df, private_df=private_df, split=split, attack_type="static"
    )
    for _, metrics in static_results.items():
        flattened_metrics["static/f1"] = metrics["eval_F1 Score"]
        flattened_metrics["static/mcc"] = metrics[
            "eval_Matthew correlation coefficient"
        ]

    return flattened_metrics


def anonymity(df, qi_cols=["keywords"], sensitive_col=["ground_texts"]):
    results = {}
    score, violating_groups = an.recursive_c_l_diversity(
        df, quasi_ident=qi_cols, sens_att=sensitive_col
    )
    results["privacy/recursive_c_l_diversity_score"] = (
        float(score) if score == score else 0.0
    )  # catch NaN
    results["privacy/recursive_c_l_diversity_violations"] = violating_groups
    results["privacy/k_anonymity"] = an.k_anonymity(df, quasi_ident=qi_cols)
    results["privacy/alpha_k_anonymity"] = an.alpha_k_anonymity(
        df, quasi_ident=qi_cols, sens_att=sensitive_col
    )[1]
    results["privacy/delta_disclosure"] = an.delta_disclosure(
        df, quasi_ident=qi_cols, sens_att=sensitive_col
    )
    results["privacy/beta_likeness"] = an.enhanced_beta_likeness(
        df, quasi_ident=qi_cols, sens_att=sensitive_col
    )
    results = {k: float(v) if v is not None else None for k, v in results.items()}
    return results
