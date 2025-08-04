from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util


class _Fid:
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

    @staticmethod
    def _compute_fid(
        real_embeddings: List[torch.Tensor], synth_embeddings: List[torch.Tensor]
    ) -> dict:
        """
        Computes the Frechet Inception Distance given a list of real and
        synthetic embedding representations of the source and reference documents.
        """
        real_avg_embedding = sum(real_embeddings) / len(real_embeddings)
        real_scores = [
            util.pytorch_cos_sim(real_avg_embedding, real_embedding)
            for real_embedding in real_embeddings
        ]
        synth_scores = [
            util.pytorch_cos_sim(real_avg_embedding, synth_embedding)
            for synth_embedding in synth_embeddings
        ]

        return {
            "real_scores": [i.item() for i in real_scores],
            "synth_scores": [i.item() for i in synth_scores],
        }

    def compute(self, references: list[str], predictions: list[str]) -> float:
        """
        Calcule la FID entre deux listes de textes.

        :param references: Textes de référence.
        :param predictions: Textes générés.
        :return: Valeur de la FID.
        """

        synth_text_embedding = self.model.encode(predictions)
        real_text_embedding = self.model.encode(references)

        fid_score_result = _Fid._compute_fid(
            real_text_embedding,
            synth_text_embedding,
        )
        return np.mean(fid_score_result["synth_scores"])
