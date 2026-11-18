import unittest

import numpy as np
import torch

from src.evaluate import fit_temperature
from src.metrics import classification_metrics
from src.model import ReverseGradient
from src.prepare import normalize


class ResearchLogicTests(unittest.TestCase):
    def test_missing_title_preserves_example_and_label(self):
        row = normalize({"review_id": "synthetic", "review_body": "A review", "review_title": None,
                         "product_category": "book", "language": "en", "stars": 5})
        self.assertEqual(row["title"], "")
        self.assertEqual(row["label"], 4)
        self.assertEqual(row["body"], "A review")

    def test_gradient_reversal(self):
        x = torch.tensor([1., 2.], requires_grad=True)
        weights = torch.tensor([3., -2.])
        out = ReverseGradient.apply(x, 0.25)
        self.assertTrue(torch.equal(out, x))
        (out * weights).sum().backward()
        torch.testing.assert_close(x.grad, -0.25 * weights)

    def test_metrics_known_values(self):
        z = np.zeros((5, 5))
        y = np.arange(5)
        m = classification_metrics(z, y)
        self.assertAlmostEqual(m["accuracy"], 0.2)
        self.assertAlmostEqual(m["mae"], 2.0)
        self.assertAlmostEqual(m["nll"], np.log(5))
        self.assertAlmostEqual(m["brier"], 0.8)
        self.assertAlmostEqual(m["ece"], 0.0)

    def test_temperature_improves_calibration_nll_preserves_predictions(self):
        z = np.tile(np.array([[8., 0., 0., 0., 0.]]), (100, 1))
        y = np.repeat(np.arange(5), 20)
        fit = fit_temperature(z, y)
        self.assertGreater(fit["temperature"], 1)
        self.assertLess(fit["calibration_nll_after"], fit["calibration_nll_before"])
        np.testing.assert_array_equal(z.argmax(1), (z / fit["temperature"]).argmax(1))


if __name__ == "__main__":
    unittest.main()
