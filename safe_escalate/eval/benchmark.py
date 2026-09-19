"""
Empirical evaluation and benchmarking suite.
Compares SafeEscalate against standard industry and research baselines:
1. Static Threshold (tau = 0.5)
2. Score-Based Dual-Threshold Abstention (tau_low=0.25, tau_high=0.75)
3. Cost-Sensitive Optimal Thresholding
4. SafeEscalate (Proposed Cascaded Uncertainty-Aware Framework)
"""

import time
import numpy as np
from typing import Dict, Any, List, Tuple
from safe_escalate.config import AppConfig
from safe_escalate.data.dataset_generator import TransactionDatasetGenerator
from safe_escalate.data.schema import Transaction
from safe_escalate.models.base_classifier import BaseFraudClassifier, Tier2EvidenceValidator
from safe_escalate.models.conformal_predictor import ConformalFraudPredictor
from safe_escalate.models.uncertainty_estimator import UncertaintyEstimator
from safe_escalate.decision.policy_engine import SafeEscalatePolicyEngine
from safe_escalate.decision.cost_matrix import CostMatrixEvaluator


class BenchmarkSuite:
    """
    Executes standardized comparative evaluations across competing fraud decision policies.
    """

    def __init__(self, config: AppConfig = None, dataset_type: str = "kaggle"):
        self.config = config or AppConfig()
        self.dataset_type = dataset_type
        self.generator = TransactionDatasetGenerator(
            random_seed=self.config.random_seed, fraud_ratio=0.07
        )
        self.cost_evaluator = CostMatrixEvaluator(self.config.costs)

    def prepare_experiment(
        self, n_samples: int = 50000, dataset_type: Optional[str] = None
    ) -> Tuple[SafeEscalatePolicyEngine, List[Transaction], Dict[str, Any]]:
        """
        Generates data (Kaggle or Synthetic), fits models, calibrates conformal bounds, and returns policy engine & test set.
        """
        ds_type = dataset_type or self.dataset_type

        if ds_type == "kaggle":
            from safe_escalate.data.kaggle_loader import KaggleDatasetLoader
            loader = KaggleDatasetLoader(random_seed=self.config.random_seed)
            splits = loader.generate_splits(max_samples=n_samples)

            X_train_base = splits["X_train"]
            y_train = splits["y_train"]
            X_cal_base = splits["X_cal"]
            y_cal = splits["y_cal"]
            X_train_full = splits["X_train_full"]
            test_txs = splits["test_transactions"]
        else:
            X_base, y, transactions = self.generator.generate_dataset(n_samples=n_samples)

            # 3-way split: Train (60%), Calibration (20%), Test (20%)
            n_train = int(n_samples * 0.60)
            n_cal = int(n_samples * 0.20)

            X_train_base, y_train = X_base[:n_train], y[:n_train]
            X_cal_base, y_cal = X_base[n_train : n_train + n_cal], y[n_train : n_train + n_cal]

            train_txs = transactions[:n_train]
            test_txs = transactions[n_train + n_cal :]

            collector_mock = [
                np.array([
                    tx.device_trust_score,
                    tx.carrier_sim_swap_age_days,
                    tx.ip_country_match,
                    tx.two_factor_auth_success,
                ], dtype=float)
                for tx in train_txs
            ]
            X_train_sec = np.vstack(collector_mock)
            X_train_full = np.hstack([X_train_base, X_train_sec])

        # 1. Fit Base Classifier
        base_clf = BaseFraudClassifier(random_state=self.config.random_seed)
        base_clf.fit(X_train_base, y_train)

        # 2. Fit Conformal Predictor on holdout calibration set
        conformal = ConformalFraudPredictor(
            base_clf, alpha=self.config.uncertainty.alpha
        )
        conformal.calibrate(X_cal_base, y_cal)

        # 3. Fit Uncertainty Estimator
        uncertainty_est = UncertaintyEstimator(n_neighbors=15)
        uncertainty_est.fit(X_train_base)

        # 4. Fit Tier 2 Validator on full feature set
        tier2_val = Tier2EvidenceValidator(random_state=self.config.random_seed)
        tier2_val.fit(X_train_full, y_train)

        # 5. Initialize Engine
        engine = SafeEscalatePolicyEngine(
            base_classifier=base_clf,
            conformal_predictor=conformal,
            uncertainty_estimator=uncertainty_est,
            tier2_validator=tier2_val,
            config=self.config,
        )

        cal_metrics = conformal.evaluate_coverage(X_cal_base, y_cal)

        return engine, test_txs, cal_metrics

    def run_benchmark(self, n_samples: int = 50000, dataset_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes full comparative simulation.
        """
        ds_type = dataset_type or self.dataset_type
        engine, test_txs, cal_metrics = self.prepare_experiment(n_samples=n_samples, dataset_type=ds_type)
        base_clf = engine.base_classifier

        # Extract test base features dynamically using engine's extract_vector
        X_test = np.vstack([engine.extract_vector(tx)[0] for tx in test_txs])
        p_fraud_test = base_clf.predict_p_fraud(X_test)
        y_test = np.array([tx.is_fraud for tx in test_txs])
        amounts = np.array([tx.amount for tx in test_txs])
        n_test = len(test_txs)

        results = {}

        # -------------------------------------------------------------
        # Baseline 1: Static Threshold (tau = 0.5)
        # -------------------------------------------------------------
        b1_fraud_loss = 0.0
        b1_friction_cost = 0.0
        b1_caught = 0
        b1_total_fraud = int(np.sum(y_test))
        b1_fp = 0

        for i in range(n_test):
            pred_fraud = 1 if p_fraud_test[i] >= 0.5 else 0
            if pred_fraud == 0 and y_test[i] == 1:
                b1_fraud_loss += amounts[i] + self.config.costs.cost_false_negative_penalty
            elif pred_fraud == 1 and y_test[i] == 0:
                b1_friction_cost += self.config.costs.cost_false_positive
                b1_fp += 1
            elif pred_fraud == 1 and y_test[i] == 1:
                b1_caught += 1

        b1_total_cost = b1_fraud_loss + b1_friction_cost
        results["Static_Threshold_0.5"] = {
            "name": "Static Threshold (tau = 0.5)",
            "total_cost": round(b1_total_cost, 2),
            "fraud_loss": round(b1_fraud_loss, 2),
            "friction_cost": round(b1_friction_cost, 2),
            "query_cost": 0.0,
            "human_cost": 0.0,
            "human_reviews": 0,
            "fraud_recall_pct": round(b1_caught / b1_total_fraud * 100, 1),
            "false_positive_count": b1_fp,
            "cost_per_tx": round(b1_total_cost / n_test, 2),
        }

        # -------------------------------------------------------------
        # Baseline 2: Dual Threshold Abstention (tau_low=0.25, tau_high=0.75)
        # Directly sends all uncertain cases to human investigation
        # -------------------------------------------------------------
        b2_fraud_loss = 0.0
        b2_friction_cost = 0.0
        b2_human_reviews = 0
        b2_caught = 0
        b2_fp = 0

        for i in range(n_test):
            p = p_fraud_test[i]
            if p < 0.25:
                # Approve
                if y_test[i] == 1:
                    b2_fraud_loss += amounts[i] + self.config.costs.cost_false_negative_penalty
            elif p > 0.75:
                # Decline
                if y_test[i] == 0:
                    b2_friction_cost += self.config.costs.cost_false_positive
                    b2_fp += 1
                else:
                    b2_caught += 1
            else:
                # Send to human review
                b2_human_reviews += 1
                # 97% human accuracy
                if y_test[i] == 1:
                    b2_caught += 1
                    b2_fraud_loss += 0.03 * (amounts[i] + self.config.costs.cost_false_negative_penalty)
                else:
                    b2_friction_cost += 0.02 * self.config.costs.cost_false_positive

        b2_human_cost = b2_human_reviews * self.config.costs.cost_human_review
        b2_total_cost = b2_fraud_loss + b2_friction_cost + b2_human_cost

        results["Dual_Threshold_Abstention"] = {
            "name": "Dual-Threshold Abstention (Direct HITL)",
            "total_cost": round(b2_total_cost, 2),
            "fraud_loss": round(b2_fraud_loss, 2),
            "friction_cost": round(b2_friction_cost, 2),
            "query_cost": 0.0,
            "human_cost": round(b2_human_cost, 2),
            "human_reviews": b2_human_reviews,
            "fraud_recall_pct": round(b2_caught / b1_total_fraud * 100, 1),
            "false_positive_count": b2_fp,
            "cost_per_tx": round(b2_total_cost / n_test, 2),
        }

        # -------------------------------------------------------------
        # Baseline 3: Cost-Sensitive Threshold
        # tau_i = C_fp / (C_fp + Amount_i + Penalty)
        # -------------------------------------------------------------
        b3_fraud_loss = 0.0
        b3_friction_cost = 0.0
        b3_caught = 0
        b3_fp = 0

        for i in range(n_test):
            tau_cost = self.config.costs.cost_false_positive / (
                self.config.costs.cost_false_positive
                + amounts[i]
                + self.config.costs.cost_false_negative_penalty
            )
            pred_fraud = 1 if p_fraud_test[i] >= tau_cost else 0
            if pred_fraud == 0 and y_test[i] == 1:
                b3_fraud_loss += amounts[i] + self.config.costs.cost_false_negative_penalty
            elif pred_fraud == 1 and y_test[i] == 0:
                b3_friction_cost += self.config.costs.cost_false_positive
                b3_fp += 1
            elif pred_fraud == 1 and y_test[i] == 1:
                b3_caught += 1

        b3_total_cost = b3_fraud_loss + b3_friction_cost
        results["Cost_Sensitive_Threshold"] = {
            "name": "Cost-Sensitive Direct Threshold",
            "total_cost": round(b3_total_cost, 2),
            "fraud_loss": round(b3_fraud_loss, 2),
            "friction_cost": round(b3_friction_cost, 2),
            "query_cost": 0.0,
            "human_cost": 0.0,
            "human_reviews": 0,
            "fraud_recall_pct": round(b3_caught / b1_total_fraud * 100, 1),
            "false_positive_count": b3_fp,
            "cost_per_tx": round(b3_total_cost / n_test, 2),
        }

        # -------------------------------------------------------------
        # Proposed Framework: SafeEscalate (Cascaded 3-Tier Policy)
        # -------------------------------------------------------------
        se_fraud_loss = 0.0
        se_friction_cost = 0.0
        se_query_cost = 0.0
        se_human_cost = 0.0
        se_human_reviews = 0
        se_tier1_resolved = 0
        se_tier0_autonomous = 0
        se_caught = 0
        se_fp = 0

        for tx in test_txs:
            packet = engine.process_transaction(tx)

            if packet.escalation_tier == 0:
                se_tier0_autonomous += 1
                if packet.final_action == "APPROVE":
                    if tx.is_fraud == 1:
                        se_fraud_loss += tx.amount + self.config.costs.cost_false_negative_penalty
                else:
                    if tx.is_fraud == 0:
                        se_friction_cost += self.config.costs.cost_false_positive
                        se_fp += 1
                    else:
                        se_caught += 1

            elif packet.escalation_tier == 1:
                se_tier1_resolved += 1
                se_query_cost += self.config.costs.cost_evidence_acquisition
                if packet.final_action.startswith("APPROVE"):
                    if tx.is_fraud == 1:
                        se_fraud_loss += tx.amount + self.config.costs.cost_false_negative_penalty
                    else:
                        # Minor 2FA friction
                        se_friction_cost += self.config.costs.cost_2fa_friction
                else:
                    if tx.is_fraud == 0:
                        se_friction_cost += self.config.costs.cost_false_positive
                        se_fp += 1
                    else:
                        se_caught += 1

            elif packet.escalation_tier == 2:
                se_human_reviews += 1
                se_query_cost += self.config.costs.cost_evidence_acquisition
                se_human_cost += self.config.costs.cost_human_review
                # 97% human accuracy
                if tx.is_fraud == 1:
                    se_caught += 1
                    se_fraud_loss += 0.03 * (tx.amount + self.config.costs.cost_false_negative_penalty)
                else:
                    se_friction_cost += 0.02 * self.config.costs.cost_false_positive

        se_total_cost = se_fraud_loss + se_friction_cost + se_query_cost + se_human_cost
        workload_reduction = (
            (b2_human_reviews - se_human_reviews) / b2_human_reviews * 100
            if b2_human_reviews > 0
            else 0.0
        )
        cost_savings_vs_static = (
            (b1_total_cost - se_total_cost) / b1_total_cost * 100
            if b1_total_cost > 0
            else 0.0
        )

        results["SafeEscalate_Proposed"] = {
            "name": "SafeEscalate (Uncertainty-Aware Cascade)",
            "total_cost": round(se_total_cost, 2),
            "fraud_loss": round(se_fraud_loss, 2),
            "friction_cost": round(se_friction_cost, 2),
            "query_cost": round(se_query_cost, 2),
            "human_cost": round(se_human_cost, 2),
            "human_reviews": se_human_reviews,
            "tier1_resolved_count": se_tier1_resolved,
            "tier0_autonomous_count": se_tier0_autonomous,
            "human_workload_reduction_pct": round(workload_reduction, 1),
            "cost_savings_pct": round(cost_savings_vs_static, 1),
            "fraud_recall_pct": round(se_caught / b1_total_fraud * 100, 1),
            "false_positive_count": se_fp,
            "cost_per_tx": round(se_total_cost / n_test, 2),
        }

        return {
            "test_sample_size": n_test,
            "total_frauds": b1_total_fraud,
            "conformal_calibration": cal_metrics,
            "models": results,
        }
