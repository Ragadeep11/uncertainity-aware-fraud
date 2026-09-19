"""
Unit tests for Sequential Adaptive Evidence Selection (VoI).
"""

import unittest
from safe_escalate.data.schema import Transaction
from safe_escalate.decision.evidence_selector import SequentialEvidenceSelector, AVAILABLE_EVIDENCE_SOURCES


class TestEvidenceSelector(unittest.TestCase):
    def setUp(self):
        self.selector = SequentialEvidenceSelector(
            uncertainty_stopping_threshold=0.22,
            max_evidence_budget=0.60,
            random_seed=42,
        )

    def test_voi_ranking(self):
        candidate_keys = list(AVAILABLE_EVIDENCE_SOURCES.keys())
        ranked = self.selector.rank_evidence(candidate_keys, current_uncertainty=0.75)
        self.assertGreater(len(ranked), 0)
        # Verify descending order of VoI scores
        scores = [item[1] for item in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_hyderabad_85000_sequential_resolution(self):
        tx = Transaction(
            transaction_id="IN-HYD-85000-TEST",
            amount=85000.0,
            customer_home_state="Andhra Pradesh",
            location_city="Hyderabad",
            time_of_day="02:13 AM",
            normal_avg_amount=2500.0,
            normal_device="Samsung Galaxy S24",
            device_fingerprint="Unknown Device",
            is_fraud=1,
        )

        p_final, u_final, trajectory, reason, resolved_auto = (
            self.selector.run_sequential_investigation(
                tx=tx, p_initial=0.72, u_initial=0.78, canonical_mode="two_step"
            )
        )

        # In Hyderabad scenario, 2 steps (Tx history + Device history) resolve it
        self.assertEqual(len(trajectory), 2)
        self.assertEqual(trajectory[0].evidence_type, "transaction_history")
        self.assertEqual(trajectory[1].evidence_type, "device_history")
        self.assertGreater(p_final, 0.90)
        self.assertLessEqual(u_final, self.selector.tau_stop)
        self.assertTrue(resolved_auto)
        self.assertIn("Stopping Rule Satisfied", reason)

    def test_inconclusive_human_escalation(self):
        tx = Transaction(
            transaction_id="TX-INCONCLUSIVE-TEST",
            amount=85000.0,
            normal_avg_amount=2500.0,
            is_fraud=1,
        )

        p_final, u_final, trajectory, reason, resolved_auto = (
            self.selector.run_sequential_investigation(
                tx=tx, p_initial=0.72, u_initial=0.78, canonical_mode="inconclusive_human"
            )
        )

        self.assertEqual(len(trajectory), 3)
        self.assertFalse(resolved_auto)
        self.assertGreater(u_final, self.selector.tau_stop)
        self.assertIn("Escalated to Human", reason)


if __name__ == "__main__":
    unittest.main()
