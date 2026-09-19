"""
Unit tests for the 5 Canonical Research Pathways in SafeEscalate.
"""

import unittest
from safe_escalate.eval.benchmark import BenchmarkSuite
from safe_escalate.data.schema import Transaction


class TestCanonicalCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        suite = BenchmarkSuite(dataset_type="synthetic")
        cls.engine, _, _ = suite.prepare_experiment(n_samples=1500, dataset_type="synthetic")

    def test_case_1_confident_genuine(self):
        tx = Transaction(
            transaction_id="TX-CASE1-TEST",
            amount=600.0,
            merchant_category=1,
            distance_from_home=2.0,
            distance_from_last_tx=1.0,
            ratio_to_median_price=0.9,
            repeat_retailer=1,
            used_chip=1,
            used_pin=1,
            online_order=0,
            velocity_1h=1,
            velocity_24h=2,
            is_fraud=0,
        )
        packet = self.engine.process_transaction(tx)
        self.assertEqual(packet.escalation_tier, 0)
        self.assertEqual(packet.final_action, "APPROVE")
        self.assertEqual(packet.canonical_case_id, "CASE_1_CONFIDENT_GENUINE")
        self.assertIsNotNone(packet.blockchain_block_hash)

    def test_case_4_hyderabad_outlier(self):
        tx = Transaction(
            transaction_id="IN-HYD-85000-TEST",
            amount=85000.0,
            merchant_category=5,
            distance_from_home=165.0,
            distance_from_last_tx=45.0,
            ratio_to_median_price=34.0,
            repeat_retailer=0,
            used_chip=0,
            used_pin=0,
            online_order=1,
            velocity_1h=1,
            velocity_24h=1,
            customer_home_state="Andhra Pradesh",
            location_city="Hyderabad",
            time_of_day="02:13 AM",
            normal_avg_amount=2500.0,
            normal_device="Samsung Galaxy S24",
            device_fingerprint="Unknown Device",
            is_fraud=1,
        )
        packet = self.engine.process_transaction(tx, canonical_mode="two_step")
        self.assertEqual(packet.escalation_tier, 1)
        self.assertIn("DECLINE", packet.final_action)
        self.assertEqual(len(packet.investigation_trajectory), 2)
        self.assertEqual(packet.canonical_case_id, "CASE_4_UNCERTAIN_TWO_STEP_RESOLVED")
        self.assertIsNotNone(packet.blockchain_block_hash)

    def test_case_5_inconclusive_human_escalation(self):
        tx = Transaction(
            transaction_id="TX-CASE5-TEST",
            amount=85000.0,
            merchant_category=5,
            distance_from_home=85.0,
            ratio_to_median_price=34.0,
            normal_avg_amount=2500.0,
            is_fraud=1,
        )
        packet = self.engine.process_transaction(tx, canonical_mode="inconclusive_human")
        self.assertEqual(packet.escalation_tier, 2)
        self.assertEqual(packet.final_action, "HUMAN_ESCALATION")
        self.assertEqual(len(packet.investigation_trajectory), 3)
        self.assertEqual(packet.canonical_case_id, "CASE_5_INCONCLUSIVE_HUMAN_ESCALATION")
        self.assertIsNotNone(packet.blockchain_block_hash)


if __name__ == "__main__":
    unittest.main()
