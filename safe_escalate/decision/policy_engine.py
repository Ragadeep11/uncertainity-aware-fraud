"""
The Core SafeEscalate Decision & Escalation Policy Engine.
Orchestrates Conformal Prediction Sets, Epistemic/Aleatoric Uncertainty,
Dynamic Micro-Evidence Acquisition, and Cost-Optimal Human Triaging.
"""

from typing import Optional, Dict, Any
import numpy as np
from safe_escalate.config import AppConfig
from safe_escalate.data.schema import Transaction, DecisionPacket, EvidencePayload
from safe_escalate.data.dataset_generator import TransactionDatasetGenerator
from safe_escalate.models.base_classifier import BaseFraudClassifier, Tier2EvidenceValidator
from safe_escalate.models.conformal_predictor import ConformalFraudPredictor
from safe_escalate.models.uncertainty_estimator import UncertaintyEstimator
from safe_escalate.decision.evidence_collector import DynamicEvidenceCollector
from safe_escalate.decision.cost_matrix import CostMatrixEvaluator


class SafeEscalatePolicyEngine:
    """
    Cascaded 3-tier uncertainty-aware decision policy.
    """

    def __init__(
        self,
        base_classifier: BaseFraudClassifier,
        conformal_predictor: ConformalFraudPredictor,
        uncertainty_estimator: UncertaintyEstimator,
        tier2_validator: Tier2EvidenceValidator,
        config: Optional[AppConfig] = None,
    ):
        self.base_classifier = base_classifier
        self.conformal_predictor = conformal_predictor
        self.uncertainty_estimator = uncertainty_estimator
        self.tier2_validator = tier2_validator
        self.config = config or AppConfig()
        self.evidence_collector = DynamicEvidenceCollector(
            cost_config=self.config.costs, random_seed=self.config.random_seed
        )
        self.cost_evaluator = CostMatrixEvaluator(self.config.costs)

    def extract_vector(self, tx: Transaction) -> Tuple[np.ndarray, list]:
        """Extracts appropriate feature vector based on whether model was trained on Kaggle or synthetic data."""
        expected_dim = getattr(self.base_classifier.scaler, "n_features_in_", 11)
        if expected_dim in (29, 30) or (tx.pca_features is not None and len(tx.pca_features) >= 28):
            if tx.pca_features:
                vec = [float(tx.pca_features.get("Time", 0.0))]
                for j in range(1, 29):
                    vec.append(float(tx.pca_features.get(f"V{j}", 0.0)))
                vec.append(float(tx.pca_features.get("Amount", tx.amount)))
            else:
                # Approximate V-vector mapping for standard transactions
                v1 = (tx.distance_from_home - 3.0) / 12.0
                v2 = (tx.distance_from_last_tx - 1.0) / 6.0
                v3 = 1.0 if tx.repeat_retailer else -1.5
                v4 = 0.5 if tx.used_chip else 2.5
                v5 = 2.0 if tx.online_order else 0.0
                v6 = float(tx.velocity_1h) * 0.5
                v7 = float(tx.velocity_24h) * 0.3
                vec = [0.0, v1, v2, v3, v4, v5, v6, v7] + [0.0] * 21 + [float(tx.amount)]
            names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
            return np.array(vec, dtype=float), names
        else:
            return TransactionDatasetGenerator.extract_base_vector(tx), TransactionDatasetGenerator.BASE_FEATURE_NAMES

    def process_transaction(self, tx: Transaction) -> DecisionPacket:
        """
        Executes the 3-Tier SafeEscalate decision cascade for a single transaction.
        """
        x_base, feature_names = self.extract_vector(tx)

        # Tier 1: Base Model Scoring
        p_initial = float(self.base_classifier.predict_p_fraud(x_base.reshape(1, -1))[0])
        conformal_set = self.conformal_predictor.predict_set(x_base)
        u_total, u_aleatoric, u_epistemic = self.uncertainty_estimator.decompose(
            x_base, p_initial
        )

        attributions = self.uncertainty_estimator.compute_feature_attributions(
            x_base, feature_names
        )

        # Ambiguity determination
        is_conformal_ambiguous = len(conformal_set) > 1
        is_high_entropy = u_aleatoric >= self.config.uncertainty.aleatoric_threshold
        is_high_epistemic = u_epistemic >= self.config.uncertainty.epistemic_threshold
        is_high_value = tx.amount >= self.config.uncertainty.high_value_amount_threshold

        is_ambiguous = is_conformal_ambiguous or is_high_entropy or is_high_epistemic

        # -------------------------------------------------------------
        # Tier 0: Direct Autonomous Decision
        # -------------------------------------------------------------
        if not is_ambiguous and not is_high_value:
            if conformal_set == ["Legit"] or p_initial < 0.20:
                action = "APPROVE"
            else:
                action = "DECLINE"

            packet = DecisionPacket(
                transaction_id=tx.transaction_id,
                amount=tx.amount,
                p_fraud_initial=round(p_initial, 4),
                p_fraud_final=round(p_initial, 4),
                conformal_set=conformal_set,
                is_ambiguous=False,
                aleatoric_uncertainty=round(u_aleatoric, 4),
                epistemic_uncertainty=round(u_epistemic, 4),
                final_action=action,
                escalation_tier=0,
                evidence_collected=None,
                investigator_rationale="Autonomous resolution: Confident conformal prediction set with negligible uncertainty.",
                feature_attributions=attributions,
                ground_truth=tx.is_fraud,
            )
            if tx.is_fraud is not None:
                packet.operational_cost = self.cost_evaluator.evaluate_realized_cost(
                    action, tx.is_fraud, tx.amount, tier=0
                )
            return packet

        # -------------------------------------------------------------
        # Tier 1: Dynamic Micro-Evidence Step-Up
        # -------------------------------------------------------------
        # If high value + severe epistemic anomaly, bypass directly to human review
        bypass_to_human = is_high_value and is_high_epistemic

        if not bypass_to_human:
            evidence = self.evidence_collector.fetch_evidence(tx)
            sec = np.array([
                evidence.device_trust_score,
                evidence.carrier_sim_swap_age_days,
                evidence.ip_country_match,
                evidence.two_factor_auth_success,
            ], dtype=float)
            x_full = np.concatenate([x_base, sec])
            p_final = float(self.tier2_validator.predict_p_fraud(x_full.reshape(1, -1))[0])

            # Check if secondary evidence collapses the uncertainty
            step_up_success = (
                evidence.two_factor_auth_success == 1
                and p_final < 0.35
                and evidence.device_trust_score > 0.40
            )
            step_up_fraud_confirmed = (
                evidence.two_factor_auth_success == 0
                or p_final > 0.80
                or (evidence.carrier_sim_swap_age_days < 7 and p_final > 0.50)
            )

            if step_up_success:
                packet = DecisionPacket(
                    transaction_id=tx.transaction_id,
                    amount=tx.amount,
                    p_fraud_initial=round(p_initial, 4),
                    p_fraud_final=round(p_final, 4),
                    conformal_set=conformal_set,
                    is_ambiguous=True,
                    aleatoric_uncertainty=round(u_aleatoric, 4),
                    epistemic_uncertainty=round(u_epistemic, 4),
                    final_action="APPROVE (STEP-UP VERIFIED)",
                    escalation_tier=1,
                    evidence_collected=evidence.model_dump(),
                    investigator_rationale="Tier-1 Resolution: Ambiguity successfully resolved via 2FA & trusted device telemetry.",
                    feature_attributions=attributions,
                    ground_truth=tx.is_fraud,
                )
                if tx.is_fraud is not None:
                    packet.operational_cost = self.cost_evaluator.evaluate_realized_cost(
                        "APPROVE", tx.is_fraud, tx.amount, tier=1
                    )
                return packet

            elif step_up_fraud_confirmed:
                packet = DecisionPacket(
                    transaction_id=tx.transaction_id,
                    amount=tx.amount,
                    p_fraud_initial=round(p_initial, 4),
                    p_fraud_final=round(p_final, 4),
                    conformal_set=conformal_set,
                    is_ambiguous=True,
                    aleatoric_uncertainty=round(u_aleatoric, 4),
                    epistemic_uncertainty=round(u_epistemic, 4),
                    final_action="DECLINE (STEP-UP FAILED)",
                    escalation_tier=1,
                    evidence_collected=evidence.model_dump(),
                    investigator_rationale="Tier-1 Resolution: Declined following failed 2FA challenge and suspicious carrier/device flags.",
                    feature_attributions=attributions,
                    ground_truth=tx.is_fraud,
                )
                if tx.is_fraud is not None:
                    packet.operational_cost = self.cost_evaluator.evaluate_realized_cost(
                        "DECLINE", tx.is_fraud, tx.amount, tier=1
                    )
                return packet

            evidence_dict = evidence.model_dump()
        else:
            evidence_dict = None
            p_final = p_initial

        # -------------------------------------------------------------
        # Tier 2: Human-in-the-Loop (HITL) Investigator Escalation
        # -------------------------------------------------------------
        reasons = []
        if is_high_value:
            reasons.append(f"High-value transaction (${tx.amount:.2f} >= ${self.config.uncertainty.high_value_amount_threshold:.0f})")
        if is_conformal_ambiguous:
            reasons.append(f"Ambiguous conformal set {conformal_set}")
        if is_high_epistemic:
            reasons.append(f"Novel OOD anomaly pattern (epistemic score {u_epistemic:.2f})")
        if not bypass_to_human:
            reasons.append("Step-up micro-evidence remained inconclusive")

        rationale = "Escalated to human review queue: " + "; ".join(reasons) + "."

        packet = DecisionPacket(
            transaction_id=tx.transaction_id,
            amount=tx.amount,
            p_fraud_initial=round(p_initial, 4),
            p_fraud_final=round(p_final, 4),
            conformal_set=conformal_set,
            is_ambiguous=True,
            aleatoric_uncertainty=round(u_aleatoric, 4),
            epistemic_uncertainty=round(u_epistemic, 4),
            final_action="HUMAN_ESCALATION",
            escalation_tier=2,
            evidence_collected=evidence_dict,
            investigator_rationale=rationale,
            feature_attributions=attributions,
            ground_truth=tx.is_fraud,
        )
        if tx.is_fraud is not None:
            packet.operational_cost = self.cost_evaluator.evaluate_realized_cost(
                "HUMAN_ESCALATION", tx.is_fraud, tx.amount, tier=2
            )
        return packet
