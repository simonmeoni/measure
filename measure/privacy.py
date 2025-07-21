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
