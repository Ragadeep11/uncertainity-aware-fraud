"""
Unit tests for ML models, Conformal Prediction coverage, and Uncertainty Decomposition.
"""

import unittest
import numpy as np
from safe_escalate.data.dataset_generator import TransactionDatasetGenerator
from safe_escalate.models.base_classifier import BaseFraudClassifier
from safe_escalate.models.conformal_predictor import ConformalFraudPredictor
from safe_escalate.models.uncertainty_estimator import UncertaintyEstimator


class TestModelsAndUncertainty(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.gen = TransactionDatasetGenerator(random_seed=42)
        X, y, txs = cls.gen.generate_dataset(n_samples=2500)

        # Split 60/20/20
        cls.X_train, cls.y_train = X[:1500], y[:1500]
        cls.X_cal, cls.y_cal = X[1500:2000], y[1500:2000]
        cls.X_test, cls.y_test = X[2000:], y[2000:]

        cls.base_clf = BaseFraudClassifier(random_state=42)
        cls.base_clf.fit(cls.X_train, cls.y_train)

        cls.conformal = ConformalFraudPredictor(cls.base_clf, alpha=0.05)
        cls.conformal.calibrate(cls.X_cal, cls.y_cal)

        cls.uq_est = UncertaintyEstimator(n_neighbors=10)
        cls.uq_est.fit(cls.X_train)

    def test_base_classifier_probabilities(self):
        probs = self.base_clf.predict_proba(self.X_test[:50])
        self.assertEqual(probs.shape, (50, 2))
        np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

    def test_conformal_coverage_guarantee(self):
        metrics = self.conformal.evaluate_coverage(self.X_test, self.y_test)
        # For alpha = 0.05, target is 95%. Due to finite sample variation on 500 test items,
        # empirical coverage should comfortably be between 92% and 98%.
        self.assertGreaterEqual(metrics["empirical_coverage"], 0.91)
        self.assertLessEqual(metrics["empirical_coverage"], 1.0)
        self.assertTrue(0.0 <= metrics["ambiguity_rate"] <= 1.0)

    def test_uncertainty_decomposition(self):
        x_sample = self.X_test[0]
        p = float(self.base_clf.predict_p_fraud(x_sample.reshape(1, -1))[0])

        u_tot, u_ale, u_epi = self.uq_est.decompose(x_sample, p)
        self.assertTrue(0.0 <= u_tot <= 1.0)
        self.assertTrue(0.0 <= u_ale <= 1.0)
        self.assertTrue(0.0 <= u_epi <= 1.0)

    def test_out_of_distribution_epistemic_increase(self):
        # A synthetic extreme outlier transaction (amount = $95,000, distance = 4,000 miles)
        outlier = np.copy(self.X_test[0])
        outlier[0] = 95000.0  # extreme amount
        outlier[2] = 4000.0   # extreme distance

        in_dist_epi = self.uq_est.calculate_epistemic(self.X_test[0])
        outlier_epi = self.uq_est.calculate_epistemic(outlier)

        self.assertGreater(outlier_epi, in_dist_epi)
        self.assertGreater(outlier_epi, 0.70)


if __name__ == "__main__":
    unittest.main()
