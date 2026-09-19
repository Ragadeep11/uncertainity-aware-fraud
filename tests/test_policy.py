"""
Unit tests for the SafeEscalate Cascaded Decision Policy.
"""

import unittest
import numpy as np
from safe_escalate.config import AppConfig
from safe_escalate.eval.benchmark import BenchmarkSuite
from safe_escalate.data.schema import Transaction


class TestPolicyEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = AppConfig()
        cls.suite = BenchmarkSuite(cls.config)
        cls.engine, cls.test_txs, _ = cls.suite.prepare_experiment(n_samples=2500)

    def test_autonomous_legitimate_resolution(self):
        # Normal, safe transaction: $25 at local repeat retailer with chip
        tx = Transaction(
            transaction_id="TX-SAFE-01",
            amount=25.0,
            merchant_category=1,
            distance_from_home=2.0,
            distance_from_last_tx=0.5,
            ratio_to_median_price=0.9,
            repeat_retailer=1,
            used_chip=1,
            used_pin=1,
            online_order=0,
            velocity_1h=0,
            velocity_24h=1,
            is_fraud=0,
        )

        packet = self.engine.process_transaction(tx)
        self.assertEqual(packet.escalation_tier, 0)
        self.assertEqual(packet.final_action, "APPROVE")
        self.assertFalse(packet.is_ambiguous)

    def test_dynamic_stepup_trigger_on_borderline(self):
        # Borderline transaction with some risk indicators: online order, higher ratio
        tx = Transaction(
            transaction_id="TX-BORDERLINE-01",
            amount=320.0,
            merchant_category=3,
            distance_from_home=40.0,
            distance_from_last_tx=15.0,
            ratio_to_median_price=2.4,
            repeat_retailer=0,
            used_chip=0,
            used_pin=0,
            online_order=1,
            velocity_1h=2,
            velocity_24h=4,
            device_trust_score=0.88,
            carrier_sim_swap_age_days=300,
            ip_country_match=1,
            two_factor_auth_success=1,
            is_fraud=0,
        )

        packet = self.engine.process_transaction(tx)
        # Should trigger Tier 1 dynamic evidence
        self.assertIn(packet.escalation_tier, [0, 1])
        if packet.escalation_tier == 1:
            self.assertIn("STEP-UP", packet.final_action)
            self.assertIsNotNone(packet.evidence_collected)

    def test_high_value_uncertain_escalates_to_human(self):
        # High value ($4,500) with ambiguity
        tx = Transaction(
            transaction_id="TX-HIGHVAL-01",
            amount=4500.0,
            merchant_category=7,
            distance_from_home=120.0,
            distance_from_last_tx=80.0,
            ratio_to_median_price=5.5,
            repeat_retailer=0,
            used_chip=0,
            used_pin=0,
            online_order=1,
            velocity_1h=3,
            velocity_24h=6,
            is_fraud=1,
        )

        packet = self.engine.process_transaction(tx)
        self.assertEqual(packet.escalation_tier, 2)
        self.assertEqual(packet.final_action, "HUMAN_ESCALATION")
        self.assertIn("High-value transaction", packet.investigator_rationale)


if __name__ == "__main__":
    unittest.main()
