import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from measure.measure import translation_metrics, semantic_metrics, privacy_metrics


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
    @patch("measure.measure.mauve")
    def test_semantic_metrics_returns_dict(self, mock_mauve):
        # Mock mauve.compute_mauve to return a predictable result
        mock_result = MagicMock()
        mock_result.mauve = 0.75
        mock_mauve.compute_mauve.return_value = mock_result

        # Test that semantic_metrics returns a dictionary with expected structure
        predictions = ["The cat is sleeping.", "It's a beautiful day."]
        references = ["A cat is resting.", "The weather is nice today."]

        result = semantic_metrics(predictions, references)

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
    result = privacy_metrics(public_data, private_data, split=[60, 20, 20])
    print("Privacy metrics executed successfully")

    # Verify result is a dictionary with expected structure
    assert isinstance(result, dict)
    assert "static/f1" in result
    assert "static/mcc" in result
    assert "adaptive/f1" in result
    assert "adaptive/mcc" in result


if __name__ == "__main__":
    unittest.main()
