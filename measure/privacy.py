import numpy as np
import pandas as pd
from pycanon import anonymity as an
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

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


def linkage_attack_tfidf(
    public_df: pd.DataFrame,
    private_df: pd.DataFrame,
    text_col: str = "keywords",
    id_col: str = "patient_id",
) -> dict:
    """Linkage via TF-IDF + 1-NN (cosine)."""
    vectorizer = TfidfVectorizer()
    X_pub = vectorizer.fit_transform(public_df[text_col])
    X_priv = vectorizer.transform(private_df[text_col])

    nn = NearestNeighbors(n_neighbors=1, metric="cosine")
    nn.fit(X_pub)

    _, idx = nn.kneighbors(X_priv)
    predicted_ids = public_df.iloc[idx.flatten()][id_col].values
    true_ids = private_df[id_col].values

    acc = float(np.mean(predicted_ids == true_ids))
    return {
        "linkage/accuracy_tfidf": acc,
        "linkage/nb_docs": len(private_df),
        "linkage/nb_correct": int(np.sum(predicted_ids == true_ids)),
    }


def linkage_attack_embeddings(
    public_df: pd.DataFrame,
    private_df: pd.DataFrame,
    model_name: str = "sentence-transformers/paraphrase-MiniLM-L6-v2",
    text_col: str = "keywords",
    id_col: str = "patient_id",
) -> dict:
    """Linkage via embeddings Sentence-Transformers + 1-NN (cosine)."""
    st_model = SentenceTransformer(model_name)
    X_pub = st_model.encode(
        public_df[text_col].tolist(),
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    X_priv = st_model.encode(
        private_df[text_col].tolist(),
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    nn = NearestNeighbors(n_neighbors=1, metric="cosine")
    nn.fit(X_pub)

    _, idx = nn.kneighbors(X_priv)
    predicted_ids = public_df.iloc[idx.flatten()][id_col].values
    true_ids = private_df[id_col].values

    acc = float(np.mean(predicted_ids == true_ids))
    tag = model_name.split("/")[-1]  # « bio-bert-base-cased » par ex.
    return {
        f"linkage/accuracy_{tag}": acc,
        "linkage/nb_docs": len(private_df),
        "linkage/nb_correct": int(np.sum(predicted_ids == true_ids)),
    }
