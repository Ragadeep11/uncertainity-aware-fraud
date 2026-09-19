"""
Generates synthetic yet statistically realistic fraud datasets with tiered feature sets
and simulated dynamic evidence acquisition.
"""

import numpy as np
from typing import Tuple, List, Dict, Any
from safe_escalate.data.schema import Transaction, EvidencePayload


class TransactionDatasetGenerator:
    """
    Generates synthetic financial transaction streams with base features (Tier 1)
    and latent dynamic secondary features (Tier 2).
    """

    BASE_FEATURE_NAMES = [
        "amount",
        "merchant_category",
        "distance_from_home",
        "distance_from_last_tx",
        "ratio_to_median_price",
        "repeat_retailer",
        "used_chip",
        "used_pin",
        "online_order",
        "velocity_1h",
        "velocity_24h",
    ]

    SECONDARY_FEATURE_NAMES = [
        "device_trust_score",
        "carrier_sim_swap_age_days",
        "ip_country_match",
        "two_factor_auth_success",
    ]

    def __init__(self, random_seed: int = 42, fraud_ratio: float = 0.06):
        self.random_seed = random_seed
        self.fraud_ratio = fraud_ratio
        self.rng = np.random.default_rng(random_seed)

    def generate_dataset(
        self, n_samples: int = 12000
    ) -> Tuple[np.ndarray, np.ndarray, List[Transaction]]:
        """
        Generates base feature matrix X, binary labels y, and list of Transaction models.
        """
        n_fraud = int(n_samples * self.fraud_ratio)
        n_legit = n_samples - n_fraud

        # --- 1. Regular Legitimate Transactions (~80% of legit) ---
        n_legit_regular = int(n_legit * 0.82)
        n_legit_ambiguous = n_legit - n_legit_regular

        reg_amount = self.rng.lognormal(mean=3.4, sigma=0.8, size=n_legit_regular)
        reg_merchant = self.rng.integers(0, 10, size=n_legit_regular)
        reg_dist_home = self.rng.exponential(scale=8.0, size=n_legit_regular)
        reg_dist_last = self.rng.exponential(scale=2.5, size=n_legit_regular)
        reg_ratio_median = self.rng.lognormal(mean=0.0, sigma=0.3, size=n_legit_regular)
        reg_repeat = self.rng.binomial(1, 0.88, size=n_legit_regular)
        reg_chip = self.rng.binomial(1, 0.85, size=n_legit_regular)
        reg_pin = self.rng.binomial(1, 0.45, size=n_legit_regular)
        reg_online = self.rng.binomial(1, 0.25, size=n_legit_regular)
        reg_vel_1h = self.rng.poisson(lam=0.6, size=n_legit_regular)
        reg_vel_24h = reg_vel_1h + self.rng.poisson(lam=2.0, size=n_legit_regular)

        reg_device_trust = np.clip(self.rng.beta(8, 2, size=n_legit_regular), 0.3, 1.0)
        reg_sim_swap_age = self.rng.integers(120, 1500, size=n_legit_regular)
        reg_ip_match = self.rng.binomial(1, 0.98, size=n_legit_regular)
        reg_2fa_success = self.rng.binomial(1, 0.98, size=n_legit_regular)

        # --- 2. Ambiguous / Borderline Legitimate (Travelers, high spenders, ~18% of legit) ---
        # Base features look risky (high distance, online, high amount), causing model uncertainty (~0.45-0.65)
        amb_amount = self.rng.lognormal(mean=5.0, sigma=1.0, size=n_legit_ambiguous)
        amb_merchant = self.rng.integers(0, 10, size=n_legit_ambiguous)
        amb_dist_home = self.rng.exponential(scale=45.0, size=n_legit_ambiguous)
        amb_dist_last = self.rng.exponential(scale=30.0, size=n_legit_ambiguous)
        amb_ratio_median = self.rng.lognormal(mean=0.8, sigma=0.6, size=n_legit_ambiguous)
        amb_repeat = self.rng.binomial(1, 0.35, size=n_legit_ambiguous)
        amb_chip = self.rng.binomial(1, 0.30, size=n_legit_ambiguous)
        amb_pin = self.rng.binomial(1, 0.10, size=n_legit_ambiguous)
        amb_online = self.rng.binomial(1, 0.75, size=n_legit_ambiguous)
        amb_vel_1h = self.rng.poisson(lam=2.0, size=n_legit_ambiguous)
        amb_vel_24h = amb_vel_1h + self.rng.poisson(lam=5.0, size=n_legit_ambiguous)

        # BUT Tier 2 dynamic evidence shows legitimate ownership (passes 2FA, verified device)
        amb_device_trust = np.clip(self.rng.beta(7, 3, size=n_legit_ambiguous), 0.25, 0.95)
        amb_sim_swap_age = self.rng.integers(90, 800, size=n_legit_ambiguous)
        amb_ip_match = self.rng.binomial(1, 0.90, size=n_legit_ambiguous)
        amb_2fa_success = self.rng.binomial(1, 0.96, size=n_legit_ambiguous)

        legit_amount = np.concatenate([reg_amount, amb_amount])
        legit_merchant = np.concatenate([reg_merchant, amb_merchant])
        legit_dist_home = np.concatenate([reg_dist_home, amb_dist_home])
        legit_dist_last = np.concatenate([reg_dist_last, amb_dist_last])
        legit_ratio_median = np.concatenate([reg_ratio_median, amb_ratio_median])
        legit_repeat = np.concatenate([reg_repeat, amb_repeat])
        legit_chip = np.concatenate([reg_chip, amb_chip])
        legit_pin = np.concatenate([reg_pin, amb_pin])
        legit_online = np.concatenate([reg_online, amb_online])
        legit_vel_1h = np.concatenate([reg_vel_1h, amb_vel_1h])
        legit_vel_24h = np.concatenate([reg_vel_24h, amb_vel_24h])

        legit_device_trust = np.concatenate([reg_device_trust, amb_device_trust])
        legit_sim_swap_age = np.concatenate([reg_sim_swap_age, amb_sim_swap_age])
        legit_ip_match = np.concatenate([reg_ip_match, amb_ip_match])
        legit_2fa_success = np.concatenate([reg_2fa_success, amb_2fa_success])

        # --- 3. Fraudulent Transactions ---
        fraud_amount = self.rng.lognormal(mean=5.4, sigma=1.2, size=n_fraud)
        fraud_merchant = self.rng.integers(0, 10, size=n_fraud)
        fraud_dist_home = self.rng.exponential(scale=60.0, size=n_fraud)
        fraud_dist_last = self.rng.exponential(scale=40.0, size=n_fraud)
        fraud_ratio_median = self.rng.lognormal(mean=1.1, sigma=0.6, size=n_fraud)
        fraud_repeat = self.rng.binomial(1, 0.22, size=n_fraud)
        fraud_chip = self.rng.binomial(1, 0.20, size=n_fraud)
        fraud_pin = self.rng.binomial(1, 0.08, size=n_fraud)
        fraud_online = self.rng.binomial(1, 0.80, size=n_fraud)
        fraud_vel_1h = self.rng.poisson(lam=2.8, size=n_fraud)
        fraud_vel_24h = fraud_vel_1h + self.rng.poisson(lam=7.0, size=n_fraud)

        # Tier 2 Latent characteristics for Fraud (fails 2FA, low trust, recent SIM swap)
        fraud_device_trust = np.clip(self.rng.beta(2, 6, size=n_fraud), 0.0, 0.70)
        fraud_sim_swap_age = self.rng.exponential(scale=10.0, size=n_fraud).astype(int)
        fraud_ip_match = self.rng.binomial(1, 0.20, size=n_fraud)
        fraud_2fa_success = self.rng.binomial(1, 0.08, size=n_fraud)

        # Stack base matrices
        X_legit_base = np.column_stack([
            legit_amount, legit_merchant, legit_dist_home, legit_dist_last,
            legit_ratio_median, legit_repeat, legit_chip, legit_pin,
            legit_online, legit_vel_1h, legit_vel_24h
        ])
        X_fraud_base = np.column_stack([
            fraud_amount, fraud_merchant, fraud_dist_home, fraud_dist_last,
            fraud_ratio_median, fraud_repeat, fraud_chip, fraud_pin,
            fraud_online, fraud_vel_1h, fraud_vel_24h
        ])

        X_base = np.vstack([X_legit_base, X_fraud_base])
        y = np.concatenate([np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)])

        # Latent secondary dictionary mapping index to Tier 2 properties
        tier2_device = np.concatenate([legit_device_trust, fraud_device_trust])
        tier2_sim = np.concatenate([legit_sim_swap_age, fraud_sim_swap_age])
        tier2_ip = np.concatenate([legit_ip_match, fraud_ip_match])
        tier2_2fa = np.concatenate([legit_2fa_success, fraud_2fa_success])

        # Shuffle synchronously
        perm = self.rng.permutation(n_samples)
        X_base = X_base[perm]
        y = y[perm]
        tier2_device = tier2_device[perm]
        tier2_sim = tier2_sim[perm]
        tier2_ip = tier2_ip[perm]
        tier2_2fa = tier2_2fa[perm]

        # Construct Transaction domain models
        transactions: List[Transaction] = []
        for i in range(n_samples):
            tx = Transaction(
                transaction_id=f"TX-{100000 + i}",
                amount=round(float(X_base[i, 0]), 2),
                merchant_category=int(X_base[i, 1]),
                distance_from_home=round(float(X_base[i, 2]), 2),
                distance_from_last_tx=round(float(X_base[i, 3]), 2),
                ratio_to_median_price=round(float(X_base[i, 4]), 2),
                repeat_retailer=int(X_base[i, 5]),
                used_chip=int(X_base[i, 6]),
                used_pin=int(X_base[i, 7]),
                online_order=int(X_base[i, 8]),
                velocity_1h=int(X_base[i, 9]),
                velocity_24h=int(X_base[i, 10]),
                # Dynamic secondary attributes pre-allocated for simulated query
                device_trust_score=round(float(tier2_device[i]), 3),
                carrier_sim_swap_age_days=int(tier2_sim[i]),
                ip_country_match=int(tier2_ip[i]),
                two_factor_auth_success=int(tier2_2fa[i]),
                is_fraud=int(y[i]),
            )
            transactions.append(tx)

        return X_base, y, transactions

    @staticmethod
    def extract_base_vector(tx: Transaction) -> np.ndarray:
        """Helper to convert transaction to numpy vector matching BASE_FEATURE_NAMES."""
        return np.array([
            tx.amount,
            tx.merchant_category,
            tx.distance_from_home,
            tx.distance_from_last_tx,
            tx.ratio_to_median_price,
            tx.repeat_retailer,
            tx.used_chip,
            tx.used_pin,
            tx.online_order,
            tx.velocity_1h,
            tx.velocity_24h,
        ], dtype=float)

    @staticmethod
    def extract_full_vector(tx: Transaction, evidence: EvidencePayload) -> np.ndarray:
        """Combines base features with dynamic Tier 2 evidence for secondary model validation."""
        base = TransactionDatasetGenerator.extract_base_vector(tx)
        sec = np.array([
            evidence.device_trust_score,
            evidence.carrier_sim_swap_age_days,
            evidence.ip_country_match,
            evidence.two_factor_auth_success,
        ], dtype=float)
        return np.concatenate([base, sec])
