"""
Unit tests for Tamper-Evident Blockchain Audit Ledger.
"""

import unittest
from safe_escalate.audit.blockchain_audit import BlockchainAuditLedger


class TestBlockchainAudit(unittest.TestCase):
    def setUp(self):
        self.ledger = BlockchainAuditLedger()

    def test_genesis_block(self):
        self.assertEqual(len(self.ledger.chain), 1)
        genesis = self.ledger.chain[0]
        self.assertEqual(genesis.index, 0)
        self.assertEqual(genesis.prev_hash, "0" * 64)
        report = self.ledger.verify_chain_integrity()
        self.assertTrue(report["valid"])

    def test_record_investigation_and_chaining(self):
        b1 = self.ledger.record_investigation(
            transaction_id="TX-TEST-001",
            amount=150.0,
            initial_p_fraud=0.01,
            initial_uncertainty=0.05,
            final_p_fraud=0.01,
            final_uncertainty=0.05,
            final_action="APPROVE",
            escalation_tier=0,
            investigation_trajectory=[],
        )
        self.assertEqual(b1.index, 1)
        self.assertEqual(b1.prev_hash, self.ledger.chain[0].block_hash)

        b2 = self.ledger.record_investigation(
            transaction_id="TX-TEST-002",
            amount=85000.0,
            initial_p_fraud=0.72,
            initial_uncertainty=0.78,
            final_p_fraud=0.96,
            final_uncertainty=0.18,
            final_action="DECLINE (STEP-UP CONFIRMED)",
            escalation_tier=1,
            investigation_trajectory=[],
        )
        self.assertEqual(b2.index, 2)
        self.assertEqual(b2.prev_hash, b1.block_hash)

        report = self.ledger.verify_chain_integrity()
        self.assertTrue(report["valid"])
        self.assertEqual(report["total_blocks"], 3)

    def test_tamper_detection(self):
        self.ledger.record_investigation(
            transaction_id="TX-AUDIT-001",
            amount=2500.0,
            initial_p_fraud=0.88,
            initial_uncertainty=0.30,
            final_p_fraud=0.92,
            final_uncertainty=0.15,
            final_action="DECLINE",
            escalation_tier=1,
            investigation_trajectory=[],
        )

        tamper_report = self.ledger.simulate_tampering(1, fake_action="TAMPERED_APPROVE")
        self.assertTrue(tamper_report["tamper_detected"])

        integrity = self.ledger.verify_chain_integrity()
        self.assertFalse(integrity["valid"])
        self.assertEqual(integrity["tampered_block_index"], 1)

        # Restore authentic block
        self.ledger.restore_block(1, tamper_report["original_action"])
        restored_integrity = self.ledger.verify_chain_integrity()
        self.assertTrue(restored_integrity["valid"])


if __name__ == "__main__":
    unittest.main()
