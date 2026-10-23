"""
Unit tests for ARGUS Baseline Model and EVSI Decision Theory.
Verifies IEEE Access 2026 published claims, empirical Hybrid Transformer reproduction,
and Net Value of Information calculations.
"""

import unittest
import numpy as np
from safe_escalate.models.argus_baseline import (
    ARGUSPublishedReference,
    ARGUSHybridTransformer,
)
from safe_escalate.decision.evidence_selector import (
    SequentialEvidenceSelector,
    AVAILABLE_EVIDENCE_SOURCES,
)
from safe_escalate.config import CostConfig


class TestARGUSBaselineAndEVSI(unittest.TestCase):
    def test_argus_published_claims(self):
        ref = ARGUSPublishedReference()
        self.assertEqual(ref.auprc, 0.880)
        self.assertEqual(ref.auroc, 0.975)
        self.assertEqual(ref.f1_score, 0.860)
        self.assertEqual(ref.precision, 0.910)
        self.assertEqual(ref.recall, 0.820)
        self.assertEqual(ref.coverage, 0.880)
        self.assertEqual(ref.selective_risk, 0.015)
        self.assertIn("Nayak", ref.authors)
        self.assertIn("10.1109/ACCESS.2026.3694614", ref.doi)

    def test_argus_hybrid_transformer_fit_and_mc_inference(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, size=(120, 10))
        y = (rng.uniform(0, 1, size=120) > 0.85).astype(int)

        model = ARGUSHybridTransformer(
            d_model=16, n_heads=2, dropout_rate=0.20, n_mc_passes=10, random_state=42
        )
        model.fit(X, y, epochs=4, batch_size=32)
        self.assertTrue(model.is_fitted)

        # MC Dropout inference
        mean_probs, mc_var = model.predict_p_fraud_mc(X[:10])
        self.assertEqual(len(mean_probs), 10)
        self.assertEqual(len(mc_var), 10)
        self.assertTrue(np.all(mean_probs >= 0.0) and np.all(mean_probs <= 1.0))
        self.assertTrue(np.all(mc_var >= 0.0))

        # Calibrate selective prediction at 88% target coverage
        model.calibrate_selective_threshold(X[:50], target_coverage=0.88)
        self.assertGreater(model.uncertainty_threshold, 0.0)

        # Selective evaluation
        metrics = model.evaluate_selective(X[50:], y[50:], target_coverage=0.88)
        self.assertIn("coverage", metrics)
        self.assertIn("selective_risk", metrics)
        self.assertIn("brier_score", metrics)
        self.assertIn("ece", metrics)
        self.assertAlmostEqual(metrics["coverage"], 0.88, delta=0.08)

    def test_net_value_of_sample_information(self):
        selector = SequentialEvidenceSelector()
        cost_cfg = CostConfig()

        source = AVAILABLE_EVIDENCE_SOURCES["transaction_history"]
        info = selector.calculate_net_value_of_sample_information(
            source=source,
            p=0.72,
            u=0.78,
            amount=85000.0,
            cost_config=cost_cfg,
        )

        self.assertIn("loss_before", info)
        self.assertIn("expected_loss_after", info)
        self.assertIn("net_value", info)
        self.assertIn("is_cost_beneficial", info)
        # For a high-value ₹85,000 ambiguous transaction, spend history has positive EVSI
        self.assertGreater(info["net_value"], 0.0)
        self.assertTrue(info["is_cost_beneficial"])

    def test_capacity_constrained_human_review_allocation(self):
        selector = SequentialEvidenceSelector()
        cost_cfg = CostConfig()

        candidates = [
            {"transaction_id": "TX_HIGH_VALUE", "amount": 90000.0, "p_fraud": 0.70},
            {"transaction_id": "TX_MED_VALUE", "amount": 15000.0, "p_fraud": 0.50},
            {"transaction_id": "TX_LOW_VALUE", "amount": 200.0, "p_fraud": 0.45},
            {"transaction_id": "TX_TINY_VALUE", "amount": 25.0, "p_fraud": 0.52},
        ]

        # Allocate with capacity = 2
        alloc = selector.allocate_human_review_capacity(
            candidates, capacity_limit=2, cost_config=cost_cfg
        )

        self.assertEqual(alloc["admitted_count"], 2)
        self.assertEqual(alloc["overflow_count"], 2)
        # Top saving transactions should be admitted
        admitted_ids = [c["transaction_id"] for c in alloc["admitted_to_human"]]
        self.assertIn("TX_HIGH_VALUE", admitted_ids)
        self.assertIn("TX_MED_VALUE", admitted_ids)
        # Overflow transactions fallback to Bayes autonomous action
        overflow = alloc["overflow_auto_resolved"]
        for o in overflow:
            self.assertEqual(o["review_status"], "CAPACITY_EXCEEDED_AUTONOMOUS_FALLBACK")
            self.assertIn(o["assigned_decision"], ("APPROVE", "DECLINE"))


if __name__ == "__main__":
    unittest.main()
