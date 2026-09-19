"""
Unit tests for Kaggle dataset loading, 29-feature vector extraction, and policy compatibility.
"""

import unittest
import numpy as np
from safe_escalate.data.schema import Transaction
from safe_escalate.data.kaggle_loader import KaggleDatasetLoader
from safe_escalate.decision.policy_engine import SafeEscalatePolicyEngine
from safe_escalate.models.base_classifier import BaseFraudClassifier, Tier2EvidenceValidator
from safe_escalate.models.conformal_predictor import ConformalFraudPredictor
from safe_escalate.models.uncertainty_estimator import UncertaintyEstimator


class TestKaggleIntegration(unittest.TestCase):

    def test_transaction_pca_features(self):
        pca_data = {"Time": 123.0, "Amount": 45.50}
        for i in range(1, 29):
            pca_data[f"V{i}"] = round(0.1 * i, 4)

        tx = Transaction(
            transaction_id="TEST-KAGGLE-01",
            amount=45.50,
            pca_features=pca_data,
        )

        self.assertIsNotNone(tx.pca_features)
        self.assertEqual(tx.pca_features["Amount"], 45.50)
        self.assertEqual(tx.pca_features["V1"], 0.1)

    def test_kaggle_29_feature_policy_pipeline(self):
        # Create a mock 30-feature dataset (Time, V1..V28, Amount)
        n = 100
        rng = np.random.default_rng(42)
        X_mock = rng.standard_normal((n, 30))
        X_mock[:, 0] = np.arange(n)  # Time
        X_mock[:, 29] = rng.uniform(10, 300, size=n)  # Amount
        y_mock = np.array([1 if i % 10 == 0 else 0 for i in range(n)])

        expected_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        self.assertEqual(len(expected_cols), 30)

        # Fit models on 29 features
        base_clf = BaseFraudClassifier()
        base_clf.fit(X_mock, y_mock)

        conformal = ConformalFraudPredictor(base_clf, alpha=0.05)
        conformal.calibrate(X_mock[:25], y_mock[:25])

        uq = UncertaintyEstimator(n_neighbors=5)
        uq.fit(X_mock)

        tier2 = Tier2EvidenceValidator()
        X_sec = rng.uniform(0, 1, size=(n, 4))
        tier2.fit(np.hstack([X_mock, X_sec]), y_mock)

        engine = SafeEscalatePolicyEngine(
            base_classifier=base_clf,
            conformal_predictor=conformal,
            uncertainty_estimator=uq,
            tier2_validator=tier2,
        )

        # Test transaction processing with pca_features dictionary
        tx = Transaction(
            transaction_id="TX-KAGGLE-MOCK-01",
            amount=float(X_mock[0, 28]),
            pca_features={col: float(X_mock[0, idx]) for idx, col in enumerate(expected_cols)},
        )

        packet = engine.process_transaction(tx)
        self.assertTrue(any(a in packet.final_action for a in ["APPROVE", "DECLINE", "ESCALATE"]))
        self.assertEqual(packet.transaction_id, "TX-KAGGLE-MOCK-01")
        self.assertGreaterEqual(packet.p_fraud_initial, 0.0)
        self.assertLessEqual(packet.p_fraud_initial, 1.0)


if __name__ == "__main__":
    unittest.main()
