"""
Integration tests for FastAPI endpoints (direct async execution without external httpx dependency).
"""

import unittest
import asyncio
from safe_escalate.web.app import (
    initialize_system,
    serve_index,
    get_status,
    simulate_next_stream_transaction,
    get_pending_queue,
    resolve_queue_item,
    ResolvePayload,
    process_transaction,
    STATE,
)
from safe_escalate.data.schema import Transaction


class TestAPIEndpoints(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        initialize_system(n_samples=1500)

    def test_serve_index(self):
        res = asyncio.run(serve_index())
        self.assertEqual(res.status_code, 200)
        self.assertIn("SafeEscalate", res.body.decode("utf-8"))

    def test_status_endpoint(self):
        res = asyncio.run(get_status())
        self.assertEqual(res["status"], "ONLINE")
        self.assertIn("conformal_q_hat", res)
        self.assertIn("conformal_target_coverage", res)

    def test_simulate_next_stream(self):
        packet = asyncio.run(simulate_next_stream_transaction())
        self.assertIsNotNone(packet.transaction_id)
        self.assertIn(packet.escalation_tier, [0, 1, 2])
        self.assertIsInstance(packet.conformal_set, list)

    def test_direct_predict_and_queue(self):
        # Create an ambiguous high-risk transaction that escalates
        tx = Transaction(
            transaction_id="TX-API-TEST-99",
            amount=3500.0,
            merchant_category=5,
            distance_from_home=80.0,
            distance_from_last_tx=60.0,
            ratio_to_median_price=4.0,
            repeat_retailer=0,
            used_chip=0,
            used_pin=0,
            online_order=1,
            velocity_1h=3,
            velocity_24h=7,
            device_trust_score=0.2,
            carrier_sim_swap_age_days=3,
            ip_country_match=0,
            two_factor_auth_success=0,
            is_fraud=1,
        )
        packet = asyncio.run(process_transaction(tx))
        self.assertEqual(packet.escalation_tier, 2)
        self.assertEqual(packet.final_action, "HUMAN_ESCALATION")

        # Verify it appears in pending queue
        queue_data = asyncio.run(get_pending_queue())
        pending_ids = [item["queue_item"]["transaction_id"] for item in queue_data["pending_cases"]]
        self.assertIn("TX-API-TEST-99", pending_ids)

        # Resolve the case via analyst endpoint
        resolve_res = asyncio.run(
            resolve_queue_item(
                ResolvePayload(
                    transaction_id="TX-API-TEST-99",
                    decision="DECLINE",
                    analyst_id="analyst_sarah",
                    notes="Confirmed account takeover.",
                )
            )
        )
        self.assertEqual(resolve_res["status"], "RESOLVED")
        self.assertEqual(resolve_res["case"]["human_decision"], "DECLINE")


if __name__ == "__main__":
    unittest.main()
