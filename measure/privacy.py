from typing import Dict, List, Optional

import evaluate
import numpy as np
import pandas as pd
import torch
from pycanon import anonymity as an
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def _ppl_scores(
    texts: List[str],
    model_name_or_path: str,
    batch_size: int = 8,
) -> np.ndarray:
    ppl = evaluate.load("perplexity", module_type="metric")
    # compute retourne un dict avec clé "perplexities"
    vals = ppl.compute(predictions=texts, model_id="gpt2", max_length=512)
    print(vals)
    return np.array(vals["perplexities"], dtype=float)


def mia_bb_perplexity(
    model_name_or_path: str,
    public_df: pd.DataFrame,
    private_df: pd.DataFrame,
    text_col: str = "ground_texts",
    batch_size: int = 8,
) -> Dict[str, float]:
    pub_ppl = _ppl_scores(public_df[text_col].tolist(), model_name_or_path, batch_size)
    priv_ppl = _ppl_scores(
        private_df[text_col].tolist(), model_name_or_path, batch_size
    )

    scores = np.concatenate([pub_ppl, priv_ppl])  # plus petit = membre
    labels = np.concatenate([np.ones_like(pub_ppl), np.zeros_like(priv_ppl)])
    inv_scores = -scores  # ROC → plus grand = membre

    auc = roc_auc_score(labels, inv_scores)
    ap = average_precision_score(labels, inv_scores)
    thr = np.unique(scores)
    best_acc = max(accuracy_score(labels, scores <= t) for t in thr)

    return {
        "mia_bb/auc": float(auc),
        "mia_bb/ap": float(ap),
        "mia_bb/acc_best": float(best_acc),
        "mia_bb/mean_member_ppl": float(pub_ppl.mean()),
        "mia_bb/mean_nonmember_ppl": float(priv_ppl.mean()),
    }


def mia_gb_loss(
    model,
    tokenizer,
    public_df: pd.DataFrame,
    private_df: pd.DataFrame,
    text_col: str = "ground_texts",
    max_length: Optional[int] = 128,
    device: str = "cpu",
    batch_size: int = 8,
) -> Dict[str, float]:
    model.eval().to(device)

    def _loss(texts):
        losses = []
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(
                texts[i : i + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            with torch.no_grad():
                out = model(**enc, labels=enc["input_ids"])
            L = out.loss.detach().cpu()
            losses.extend([L.numpy()])
        return np.array(losses, dtype=float)

    pub_loss = _loss(public_df[text_col].tolist())
    priv_loss = _loss(private_df[text_col].tolist())

    scores = np.concatenate([pub_loss, priv_loss])  # plus petit = membre
    labels = np.concatenate([np.ones_like(pub_loss), np.zeros_like(priv_loss)])
    inv_scr = -scores

    auc = roc_auc_score(labels, inv_scr)
    ap = average_precision_score(labels, inv_scr)
    best_acc = max(accuracy_score(labels, scores <= t) for t in np.unique(scores))

    return {
        "mia_gb/auc": float(auc),
        "mia_gb/ap": float(ap),
        "mia_gb/acc_best": float(best_acc),
        "mia_gb/mean_member_loss": float(pub_loss.mean()),
        "mia_gb/mean_nonmember_loss": float(priv_loss.mean()),
    }


def membership_attack(
    model_name_or_path: str,
    public_df: pd.DataFrame,
    private_df: pd.DataFrame,
    text_col: str = "ground_texts",
    max_length: int = 128,
    batch_size: int = 8,
    device: str = "cpu",
) -> Dict[str, float]:
    # BB via evaluate
    bb = mia_bb_perplexity(
        model_name_or_path, public_df, private_df, text_col, batch_size
    )

    # GB : on charge modèle & tokenizer une seule fois
    tok = AutoTokenizer.from_pretrained(model_name_or_path)
    tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    gb = mia_gb_loss(
        model=mdl,
        tokenizer=tok,
        public_df=public_df,
        private_df=private_df,
        text_col=text_col,
        max_length=max_length,
        device=device,
        batch_size=batch_size,
    )

    res = {**bb, **gb}
    return res
