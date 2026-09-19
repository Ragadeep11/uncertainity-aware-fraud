"""
Dynamic evidence acquisition service for Tier-2 step-up verification.
"""

import time
import numpy as np
from safe_escalate.data.schema import Transaction, EvidencePayload
from safe_escalate.config import CostConfig


class DynamicEvidenceCollector:
    """
    Manages Tier-2 secondary feature acquisition (device risk SDK, carrier SIM swap, 2FA challenge).
    """

    def __init__(self, cost_config: CostConfig = None, random_seed: int = 42):
        self.config = cost_config or CostConfig()
        self.rng = np.random.default_rng(random_seed)

    def fetch_evidence(self, transaction: Transaction) -> EvidencePayload:
        """
        Retrieves secondary features for a transaction.
        Uses transaction's latent fields if populated, or simulates a live probe.
        """
        # If pre-populated (simulation / dataset context)
        if transaction.device_trust_score is not None:
            device_trust = transaction.device_trust_score
            sim_swap_age = (
                transaction.carrier_sim_swap_age_days
                if transaction.carrier_sim_swap_age_days is not None
                else 365
            )
            ip_match = (
                transaction.ip_country_match
                if transaction.ip_country_match is not None
                else 1
            )
            two_factor = (
                transaction.two_factor_auth_success
                if transaction.two_factor_auth_success is not None
                else 1
            )
        else:
            # Dynamic simulation for ad-hoc transactions entered through the API/UI
            is_risky = transaction.distance_from_home > 30 or transaction.amount > 500
            if is_risky:
                device_trust = float(np.clip(self.rng.beta(2, 5), 0.1, 0.9))
                sim_swap_age = int(self.rng.integers(1, 45))
                ip_match = int(self.rng.binomial(1, 0.4))
                two_factor = int(self.rng.binomial(1, 0.2))
            else:
                device_trust = float(np.clip(self.rng.beta(7, 2), 0.2, 0.99))
                sim_swap_age = int(self.rng.integers(90, 800))
                ip_match = int(self.rng.binomial(1, 0.95))
                two_factor = int(self.rng.binomial(1, 0.98))

        return EvidencePayload(
            device_trust_score=round(float(device_trust), 3),
            carrier_sim_swap_age_days=int(sim_swap_age),
            ip_country_match=int(ip_match),
            two_factor_auth_success=int(two_factor),
            evidence_cost=self.config.cost_evidence_acquisition,
            query_latency_ms=int(self.rng.integers(40, 180)),
        )
