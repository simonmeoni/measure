import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import torch

from measure.downstream_tasks import caml_classification, gpt_eval, mimic_iii_icd_classification
from measure.privacy import (
    anonymity,
    author_attack,
    linkage_attack_embeddings,
    linkage_attack_tfidf,
    membership_attack,
)
from measure.semantic import similarity_metrics, translation_metrics


class TestTranslationMetrics(unittest.TestCase):
    def test_translation_metrics_precise_values(self):
        # Test with known inputs to get precise expected values
        predictions = ["The cat is on the mat.", "Hello world."]
        references = ["The cat sits on the mat.", "Hello world!"]

        result = translation_metrics(predictions, references)

        # Check that all expected keys are present
        self.assertIn("rouge1", result)
        self.assertIn("rouge2", result)
        self.assertIn("rougeL", result)
        self.assertIn("bleu", result)

        # Test with exact values - these are deterministic for this input
        self.assertAlmostEqual(result["rouge1"], 0.9, places=1)
        self.assertAlmostEqual(result["rouge2"], 0.8, places=1)
        self.assertAlmostEqual(result["rougeL"], 0.9, places=1)
        self.assertTrue(isinstance(result["bleu"], (int, float)))
        self.assertAlmostEqual(result["bleu"], 45.18, places=1)

    def test_translation_metrics_perfect_match(self):
        # Test with identical predictions and references
        predictions = ["Hello world.", "The quick brown fox."]
        references = ["Hello world.", "The quick brown fox."]

        result = translation_metrics(predictions, references)

        # Perfect matches should have high scores
        self.assertEqual(result["rouge1"], 1.0)
        self.assertEqual(result["rouge2"], 1.0)
        self.assertEqual(result["rougeL"], 1.0)
        self.assertAlmostEqual(result["bleu"], 100.0)


class TestSemanticMetrics(unittest.TestCase):
    @patch("measure.semantic.mauve")
    def test_similarity_metrics_returns_dict(self, mock_mauve):
        # Mock mauve.compute_mauve to return a predictable result
        mock_result = MagicMock()
        mock_result.mauve = 0.75
        mock_mauve.compute_mauve.return_value = mock_result

        # Test that similarity_metrics returns a dictionary with expected structure
        predictions = ["The cat is sleeping.", "It's a beautiful day."]
        references = ["A cat is resting.", "The weather is nice today."]

        result = similarity_metrics(predictions, references)

        # Check that result is a dictionary
        self.assertIsInstance(result, dict)

        # Check that all expected keys are present
        self.assertIn("mauve", result)
        self.assertIn("predictions_perplexity", result)
        self.assertIn("references_perplexity", result)
        self.assertIn("fid", result)

        # Check that values are numeric
        self.assertIsInstance(result["mauve"], (int, float))
        self.assertIsInstance(result["predictions_perplexity"], (int, float))
        self.assertIsInstance(result["references_perplexity"], (int, float))
        self.assertIsInstance(result["fid"], (int, float))

        # Check that values are in reasonable ranges
        self.assertGreaterEqual(result["mauve"], 0)
        self.assertLessEqual(result["mauve"], 1)
        self.assertGreater(result["predictions_perplexity"], 0)
        self.assertGreater(result["references_perplexity"], 0)


def test_privacy_metrics():
    """Test privacy metrics functionality with 10 data points"""
    # Create sample dataframes with 10 data points for testing privacy metrics
    public_data = pd.DataFrame(
        {
            "text": [
                "This is public data",
                "Another public record",
                "Public info available",
                "Open data source",
                "General information shared",
                "Common knowledge base",
                "Publicly accessible content",
                "Open source material",
                "Community shared data",
                "Standard public record",
            ],
            "private_id": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    private_data = pd.DataFrame(
        {
            "text": [
                "This is private data",
                "Confidential record kept",
                "Secret info protected",
                "Hidden data secured",
                "Personal information stored",
                "Private knowledge preserved",
                "Confidential content maintained",
                "Secure material protected",
                "Individual private data",
                "Personal record secured",
            ]
        }
    )

    # Test privacy metrics with 10 data points
    result = author_attack(public_data, private_data, split=[60, 20, 20])
    print("Privacy metrics executed successfully")

    # Verify result is a dictionary with expected structure
    assert isinstance(result, dict)
    assert "static/f1" in result
    assert "static/mcc" in result
    assert "adaptive/f1" in result
    assert "adaptive/mcc" in result

    df = pd.DataFrame(
        {
            "keywords": [
                "diabetes hypertension",
                "hypertension diabetes",
                "asthma",
                "asthma",
                "diabetes",
            ],
            "ground_texts": [
                "Patient has diabetes and hypertension.",
                "Hypertension and diabetes both present.",
                "Asthma symptoms observed.",
                "Asthma symptoms recurring.",
                "Diabetic condition confirmed.",
            ],
        }
    )
    results = anonymity(df)

    assert isinstance(results, dict)
    for key in [
        "privacy/k_anonymity",
        "privacy/alpha_k_anonymity",
        "privacy/delta_disclosure",
        "privacy/beta_likeness",
        "privacy/recursive_c_l_diversity_score",
        "privacy/recursive_c_l_diversity_violations",
    ]:
        print(results)
        assert key in results
        assert isinstance(results[key], (int, float))

    df = pd.DataFrame(
        {
            "patient_id": [1, 1, 2, 2, 3],
            "keywords": [
                "diabetes hypertension",
                "hypertension diabetes",
                "asthma",
                "asthma copd",
                "diabetes",
            ],
        }
    )

    public_df = df.iloc[[0, 2, 4]].reset_index(drop=True)
    private_df = df.iloc[[1, 3, 0]].reset_index(drop=True)

    tfidf_res = linkage_attack_tfidf(public_df, private_df)
    assert "linkage/accuracy_tfidf" in tfidf_res
    assert 0.0 <= tfidf_res["linkage/accuracy_tfidf"] <= 1.0

    emb_res = linkage_attack_embeddings(
        public_df,
        private_df,
    )
    key = "linkage/accuracy_paraphrase-MiniLM-L6-v2"
    assert key in emb_res
    assert 0.0 <= emb_res[key] <= 1.0

    model_id = "sshleifer/tiny-gpt2"  # 14 MB, parfait pour CI
    pub = pd.DataFrame({"ground_texts": ["hello world"] * 3})
    prv = pd.DataFrame({"ground_texts": ["new unseen note"] * 2})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = membership_attack(
        model_name_or_path=model_id,
        public_df=pub,
        private_df=prv,
        device=device,
        batch_size=2,
    )

    # clés essentielles présentes
    for k in ["mia_bb/auc", "mia_gb/auc"]:
        assert k in metrics and isinstance(metrics[k], float)


class TestDownstreamTasks(unittest.TestCase):
    @patch("measure.utils.gpt_eval.AsyncOpenAI")
    def test_gpt_eval_alpacare(self, mock_async_openai):

        # Create a mock response object
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Tie"

        # Mock the AsyncOpenAI client and its chat.completions.create method
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_async_openai.return_value = mock_client

        # Test with toy data directory
        result = gpt_eval(
            data_dir="data/alpacare",
            max_samples=10,
            tested_model_name="data/alpacare/gpt-4",
        )

        # Verify result structure
        self.assertIsInstance(result, dict)
        # Check that some expected keys exist
        for key in result.keys():
            self.assertTrue(key.startswith("gpt_eval/"))
        assert result["gpt_eval/gpt-4"] == 0.5
        assert result["gpt_eval/claude-2"] == 0.5
        assert result["gpt_eval/gpt-3.5-turbo"] == 0.5
        assert result["gpt_eval/text-davinci-003"] == 0.5

        # Verify the API client was instantiated
        mock_async_openai().chat.completions.create.assert_awaited()


class TestMimicIIIClassification(unittest.TestCase):
    def test_mimic_iii_icd_classification_basic(self):
        """Test MIMIC-III ICD classification runs and returns expected metrics"""
        result = mimic_iii_icd_classification(
            test_data_dir="data/mimic-iii-test",
            train_data_dir="data/mimic-iii-test",
            model_name="microsoft/deberta-v3-base",
        )

        # Verify result is dict with expected keys
        self.assertIsInstance(result, dict)
        self.assertIn("mimic_iii_icd/accuracy", result)
        self.assertIn("mimic_iii_icd/f1", result)
        self.assertIn("mimic_iii_icd/precision", result)
        self.assertIn("mimic_iii_icd/recall", result)


class TestCAMLClassification(unittest.TestCase):
    def setUp(self):
        """Create toy CAML datasets for testing"""
        import os
        import tempfile

        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()

        # Create toy data with 3 samples each
        # Format: SUBJECT_ID,HADM_ID,LABELS,TEXT,HOT_LABELS
        train_data = [
            [
                1,
                100,
                "401.9;250.00",
                "Patient has hypertension and diabetes.",
                "[1, 0, 1, 0, 0]",
            ],
            [2, 101, "250.00", "Diabetic patient requires insulin.", "[0, 0, 1, 0, 0]"],
            [
                3,
                102,
                "401.9;272.4",
                "High blood pressure and high cholesterol.",
                "[1, 0, 0, 1, 0]",
            ],
        ]

        dev_data = [
            [4, 103, "401.9", "Hypertensive crisis observed.", "[1, 0, 0, 0, 0]"],
            [
                5,
                104,
                "250.00;272.4",
                "Diabetes with cholesterol issues.",
                "[0, 0, 1, 1, 0]",
            ],
            [6, 105, "414.01", "Coronary artery disease present.", "[0, 1, 0, 0, 0]"],
        ]

        test_data = [
            [
                7,
                106,
                "401.9;250.00",
                "Patient with both hypertension and diabetes.",
                "[1, 0, 1, 0, 0]",
            ],
            [8, 107, "272.4", "High cholesterol levels detected.", "[0, 0, 0, 1, 0]"],
            [
                9,
                108,
                "414.01;401.9",
                "Heart disease with high blood pressure.",
                "[1, 1, 0, 0, 0]",
            ],
        ]

        # Create CSV files
        columns = ["SUBJECT_ID", "HADM_ID", "LABELS", "TEXT", "HOT_LABELS"]

        train_df = pd.DataFrame(train_data, columns=columns)
        dev_df = pd.DataFrame(dev_data, columns=columns)
        test_df = pd.DataFrame(test_data, columns=columns)

        self.train_path = os.path.join(self.test_dir, "train.csv")
        self.dev_path = os.path.join(self.test_dir, "dev.csv")
        self.test_path = os.path.join(self.test_dir, "test.csv")

        train_df.to_csv(self.train_path, index=False)
        dev_df.to_csv(self.dev_path, index=False)
        test_df.to_csv(self.test_path, index=False)

    def tearDown(self):
        """Clean up temporary files"""
        import shutil

        shutil.rmtree(self.test_dir)

    def test_caml_classification_basic(self):
        """Test CAML classification runs and returns expected metrics"""
        result = caml_classification(
            train_data_path=self.train_path,
            dev_data_path=self.dev_path,
            test_data_path=self.test_path,
            model_name="microsoft/deberta-v3-base",
        )

        # Verify result is dict with expected keys
        self.assertIsInstance(result, dict)
        self.assertIn("caml/accuracy", result)
        self.assertIn("caml/f1", result)
        self.assertIn("caml/precision", result)
        self.assertIn("caml/recall", result)

        print("CAML Classification Results:", result)


if __name__ == "__main__":
    unittest.main()
