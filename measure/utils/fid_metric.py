import numpy as np
from scipy.linalg import sqrtm
from sentence_transformers import SentenceTransformer


class Fid:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        normalize: bool = True,
    ):
        """
        Initialise le calculateur de FID avec Sentence-BERT.

        :param model_name: Nom du modèle SentenceTransformer.
        :param batch_size: Taille des batchs d'encodage.
        :param normalize: Si True, les embeddings sont normalisés.
        """
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.normalize = normalize

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )

    @staticmethod
    def _compute_stats(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu = np.mean(embeddings, axis=0)
        sigma = np.cov(embeddings, rowvar=False)
        return mu, sigma

    @staticmethod
    def _compute_fid(
        mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray
    ) -> float:
        diff = mu1 - mu2
        covmean = sqrtm(sigma1 @ sigma2)
        if np.iscomplexobj(covmean):  # Nettoyage numérique
            covmean = covmean.real
        return np.sum(diff**2) + np.trace(sigma1 + sigma2 - 2 * covmean)

    def compute(self, references: list[str], predictions: list[str]) -> float:
        """
        Calcule la FID entre deux listes de textes.

        :param references: Textes de référence.
        :param predictions: Textes générés.
        :return: Valeur de la FID.
        """
        ref_embeddings = self._embed(references)
        pred_embeddings = self._embed(predictions)

        mu_ref, sigma_ref = self._compute_stats(ref_embeddings)
        mu_pred, sigma_pred = self._compute_stats(pred_embeddings)

        return self._compute_fid(mu_ref, sigma_ref, mu_pred, sigma_pred)
