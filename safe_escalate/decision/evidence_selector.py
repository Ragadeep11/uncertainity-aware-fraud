"""
Sequential Adaptive Evidence Selection Engine.
Selects optimal investigative actions based on Value of Information (VoI)
and Expected Uncertainty Reduction per unit cost.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from safe_escalate.data.schema import Transaction, InvestigationStep


class EvidenceSource:
    """Represents a discrete source of secondary evidence with economic cost and latency."""

    def __init__(
        self,
        key: str,
        name: str,
        cost: float,
        latency_ms: int,
        expected_uncertainty_reduction: float,
        description: str,
    ):
        self.key = key
        self.name = name
        self.cost = cost
        self.latency_ms = latency_ms
        self.expected_uncertainty_reduction = expected_uncertainty_reduction
        self.description = description


AVAILABLE_EVIDENCE_SOURCES: Dict[str, EvidenceSource] = {
    "transaction_history": EvidenceSource(
        key="transaction_history",
        name="Transaction Spend History",
        cost=0.05,
        latency_ms=15,
        expected_uncertainty_reduction=0.35,
        description="Checks deviation against 90-day cardholder baseline spend distribution.",
    ),
    "device_history": EvidenceSource(
        key="device_history",
        name="Device Fingerprint & SIM History",
        cost=0.10,
        latency_ms=35,
        expected_uncertainty_reduction=0.30,
        description="Evaluates hardware identifier, browser fingerprint, and carrier SIM swap recency.",
    ),
    "location_history": EvidenceSource(
        key="location_history",
        name="Location & Geo-Velocity Context",
        cost=0.15,
        latency_ms=45,
        expected_uncertainty_reduction=0.25,
        description="Calculates impossible physical velocity and anomalous timezone activity.",
    ),
    "merchant_history": EvidenceSource(
        key="merchant_history",
        name="Merchant Category & Risk Investigation",
        cost=0.20,
        latency_ms=60,
        expected_uncertainty_reduction=0.25,
        description="Analyzes merchant MCC chargeback ratios and cardholder purchase familiarity.",
    ),
    "behavioral_stepup_2fa": EvidenceSource(
        key="behavioral_stepup_2fa",
        name="Active SMS / App 2FA Step-Up Challenge",
        cost=0.30,
        latency_ms=120,
        expected_uncertainty_reduction=0.45,
        description="Dispatches interactive cryptographic one-time authentication challenge.",
    ),
}


from safe_escalate.config import CostConfig


class SequentialEvidenceSelector:
    """
    Orchestrates adaptive, multi-step evidence gathering.
    Applies Value of Information (VoI): Score(e) = E[ΔU(e)] / Cost(e)
    and Expected Value of Sample Information:
    Net Value(e) = Expected Loss Before - E[Expected Loss After(e)] - Cost(e).
    """

    def __init__(
        self,
        uncertainty_stopping_threshold: float = 0.22,
        max_evidence_budget: float = 0.60,
        min_uncertainty_gain: float = 0.04,
        random_seed: int = 42,
    ):
        self.tau_stop = uncertainty_stopping_threshold
        self.max_budget = max_evidence_budget
        self.min_gain = min_uncertainty_gain
        self.rng = np.random.default_rng(random_seed)

    @staticmethod
    def compute_immediate_loss(
        p: float, amount: float, cost_config: Optional[CostConfig] = None
    ) -> Tuple[float, str]:
        """
        Computes immediate expected loss under optimal Bayes action without further information:
        L_approve(p, A) = p * (A + C_penalty)
        L_decline(p, A) = (1 - p) * (C_false_positive + 0.02 * A)
        L*(p, A) = min(L_approve, L_decline)
        """
        cfg = cost_config or CostConfig()
        c_fp = cfg.cost_false_positive + 0.02 * amount
        loss_approve = p * (amount + cfg.cost_false_negative_penalty)
        loss_decline = (1.0 - p) * c_fp
        if loss_approve <= loss_decline:
            return float(loss_approve), "APPROVE"
        else:
            return float(loss_decline), "DECLINE"

    @staticmethod
    def compute_expected_loss_human(
        p: float, amount: float, cost_config: Optional[CostConfig] = None
    ) -> float:
        """
        Computes expected operational cost if transaction is escalated to human review:
        L_human = C_human + 0.03 * p * (A + C_penalty) + 0.02 * (1 - p) * (C_false_positive + 0.02 * A)
        """
        cfg = cost_config or CostConfig()
        c_fp = cfg.cost_false_positive + 0.02 * amount
        loss = (
            cfg.cost_human_review
            + 0.03 * p * (amount + cfg.cost_false_negative_penalty)
            + 0.02 * (1.0 - p) * c_fp
        )
        return float(loss)

    def calculate_net_value_of_sample_information(
        self,
        source: EvidenceSource,
        p: float,
        u: float,
        amount: float,
        cost_config: Optional[CostConfig] = None,
    ) -> Dict[str, Any]:
        """
        Calculates the Expected Value of Sample Information (EVSI) and Net Value:
        Net Value(e) = Expected Loss Before - E[Expected Loss After(e)] - Cost(e).
        Under uncertainty-aware decision theory:
        Loss(p, u, A) = BayesDecisionLoss(p, A) + λ * u * min(A, 5000.0)
        Evidence is cost-beneficial if and only if Net Value(e) > 0.
        """
        cfg = cost_config or CostConfig()
        lambda_risk = 0.02  # Risk premium weight per unit epistemic uncertainty
        risk_exposure = lambda_risk * u * min(amount, 5000.0)

        bayes_loss_before, action_before = self.compute_immediate_loss(p, amount, cfg)
        loss_before = bayes_loss_before + risk_exposure

        # Posterior uncertainty and branch shifts
        reduction = source.expected_uncertainty_reduction
        u_after = max(0.04, u * (1.0 - reduction))
        risk_exposure_after = lambda_risk * u_after * min(amount, 5000.0)

        shift = min(0.40, reduction * 0.90)
        p_pos = min(0.99, p + shift * (1.0 - p))
        p_neg = max(0.01, p - shift * p)

        denom = max(1e-6, p_pos - p_neg)
        q_pos = float(np.clip((p - p_neg) / denom, 0.05, 0.95))
        q_neg = 1.0 - q_pos

        loss_pos_bayes, _ = self.compute_immediate_loss(p_pos, amount, cfg)
        loss_neg_bayes, _ = self.compute_immediate_loss(p_neg, amount, cfg)
        expected_bayes_after = q_pos * loss_pos_bayes + q_neg * loss_neg_bayes
        expected_loss_after = expected_bayes_after + risk_exposure_after

        expected_loss_reduction = max(0.0, loss_before - expected_loss_after)
        net_value = expected_loss_reduction - source.cost

        return {
            "evidence_key": source.key,
            "evidence_name": source.name,
            "cost": source.cost,
            "loss_before": round(loss_before, 4),
            "expected_loss_after": round(expected_loss_after, 4),
            "expected_loss_reduction": round(expected_loss_reduction, 4),
            "net_value": round(net_value, 4),
            "is_cost_beneficial": bool(net_value > 0.0),
            "action_before": action_before,
        }

    def rank_evidence_by_net_value(
        self,
        candidate_keys: List[str],
        p: float,
        u: float,
        amount: float,
        cost_config: Optional[CostConfig] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Ranks candidate evidence sources by descending Net Value:
        Net Value(e) = ΔLoss - Cost(e)
        """
        cfg = cost_config or CostConfig()
        scored = []
        for key in candidate_keys:
            if key in AVAILABLE_EVIDENCE_SOURCES:
                source = AVAILABLE_EVIDENCE_SOURCES[key]
                info = self.calculate_net_value_of_sample_information(
                    source, p, u, amount, cfg
                )
                scored.append((key, info["net_value"], info))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    @staticmethod
    def allocate_human_review_capacity(
        candidates: List[Dict[str, Any]],
        capacity_limit: int,
        cost_config: Optional[CostConfig] = None,
    ) -> Dict[str, Any]:
        """
        Routes transactions to human review subject to a finite human queue capacity.
        Candidate transactions are prioritized by expected risk exposure savings from human review:
        Savings = Amount * (Error_auto - Error_human) * Uncertainty - C_human.

        - The top-K transactions with positive savings are routed to Human Review.
        - The remaining overflow transactions are automatically resolved via Bayes optimal action.
        """
        cfg = cost_config or CostConfig()
        scored_candidates = []
        for c in candidates:
            p = float(c.get("p_fraud", 0.5))
            amount = float(c.get("amount", 100.0))
            u = float(c.get("uncertainty", min(1.0, 4.0 * p * (1.0 - p))))

            # Automated error rate under Bayes decision
            auto_error = min(p, 1.0 - p)
            human_error = 0.03  # Human review achieves 97% accuracy
            error_reduction = max(0.0, auto_error - human_error)

            # Expected loss savings from human intervention
            savings = amount * error_reduction * u - cfg.cost_human_review

            loss_auto, auto_action = SequentialEvidenceSelector.compute_immediate_loss(p, amount, cfg)
            loss_human = SequentialEvidenceSelector.compute_expected_loss_human(p, amount, cfg)

            item = dict(c)
            item["loss_auto"] = round(loss_auto, 4)
            item["loss_human"] = round(loss_human, 4)
            item["expected_savings_from_human"] = round(savings, 4)
            item["fallback_autonomous_action"] = auto_action
            scored_candidates.append(item)

        # Sort by expected savings descending (highest value of human attention first)
        scored_candidates.sort(key=lambda x: x["expected_savings_from_human"], reverse=True)

        admitted_to_human = scored_candidates[:max(0, capacity_limit)]
        overflow_auto_resolved = scored_candidates[max(0, capacity_limit):]

        for item in admitted_to_human:
            item["assigned_decision"] = "HUMAN_ESCALATION"
            item["review_status"] = "ADMITTED_TO_QUEUE"

        for item in overflow_auto_resolved:
            item["assigned_decision"] = item["fallback_autonomous_action"]
            item["review_status"] = "CAPACITY_EXCEEDED_AUTONOMOUS_FALLBACK"

        return {
            "total_candidates": len(candidates),
            "capacity_limit": capacity_limit,
            "admitted_count": len(admitted_to_human),
            "overflow_count": len(overflow_auto_resolved),
            "admitted_to_human": admitted_to_human,
            "overflow_auto_resolved": overflow_auto_resolved,
            "capacity_utilization_pct": round(len(admitted_to_human) / max(1, capacity_limit) * 100, 1),
        }

    def calculate_voi(
        self, source: EvidenceSource, current_uncertainty: float
    ) -> float:
        """
        Calculates Value of Information / Uncertainty Reduction per Unit Cost:
        VoI(e) = E[ΔU(e)] / Cost(e)
        """
        achievable_reduction = min(
            source.expected_uncertainty_reduction, current_uncertainty * 0.75
        )
        return achievable_reduction / max(0.01, source.cost)

    def rank_evidence(
        self,
        candidate_keys: List[str],
        current_uncertainty: float,
    ) -> List[Tuple[str, float]]:
        """Ranks unqueried evidence by descending Value of Information (VoI)."""
        scored = []
        for key in candidate_keys:
            if key in AVAILABLE_EVIDENCE_SOURCES:
                source = AVAILABLE_EVIDENCE_SOURCES[key]
                voi = self.calculate_voi(source, current_uncertainty)
                scored.append((key, voi))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def simulate_query_evidence(
        self,
        evidence_key: str,
        tx: Transaction,
        p_current: float,
        u_current: float,
        force_outcome: Optional[str] = None,
    ) -> Tuple[float, float, Dict[str, Any], str]:
        """
        Simulates retrieving specific evidence for a transaction.
        Returns: (p_after, u_after, findings_dict, findings_summary)
        """
        findings = {}
        summary = ""

        is_hyderabad_scenario = (
            tx.amount >= 50000
            or (tx.customer_home_state == "Andhra Pradesh" and tx.location_city == "Hyderabad")
            or "85000" in tx.transaction_id
            or (tx.normal_avg_amount is not None and tx.amount > tx.normal_avg_amount * 10)
        )

        if evidence_key == "transaction_history":
            avg_amt = tx.normal_avg_amount or 2500.0
            ratio = round(tx.amount / avg_amt, 1)
            findings = {
                "historical_avg_amount": avg_amt,
                "current_amount": tx.amount,
                "spend_deviation_ratio": f"{ratio}x normal",
                "prior_recent_transactions": [800, 1200, 2400, 3100, 1800] if is_hyderabad_scenario else [150, 220, 85, 310],
            }
            if ratio > 10.0 or is_hyderabad_scenario:
                p_after = min(0.99, p_current + 0.16)
                u_after = max(0.05, u_current * 0.60)
                summary = f"Severe spend deviation: Current amount is {ratio}x normal cardholder baseline (avg ₹{avg_amt:,.0f})."
            elif ratio < 2.0:
                p_after = max(0.01, p_current - 0.20)
                u_after = max(0.05, u_current * 0.50)
                summary = f"Consistent with historical patterns: {ratio}x average spend."
            else:
                p_after = p_current + 0.05
                u_after = max(0.05, u_current * 0.75)
                summary = f"Moderate spend deviation ({ratio}x normal)."

        elif evidence_key == "device_history":
            normal_dev = tx.normal_device or "Samsung Galaxy S24"
            curr_dev = tx.device_fingerprint or "Unknown Device (Brand New Hardware)"
            is_new_device = (
                is_hyderabad_scenario
                or curr_dev != normal_dev
                or (tx.device_trust_score is not None and tx.device_trust_score < 0.40)
            )
            findings = {
                "registered_device": normal_dev,
                "current_device": curr_dev,
                "is_first_time_device": is_new_device,
                "carrier_sim_swap_days": tx.carrier_sim_swap_age_days if tx.carrier_sim_swap_age_days is not None else (4 if is_hyderabad_scenario else 412),
            }
            if is_new_device:
                p_after = min(0.99, p_current + 0.12)
                u_after = max(0.05, u_current * 0.45)
                summary = f"Unrecognized hardware fingerprint: Never seen on account. Recent SIM change detected."
            else:
                p_after = max(0.01, p_current - 0.25)
                u_after = max(0.05, u_current * 0.40)
                summary = f"Hardware verified: Matches registered {normal_dev} with unbroken SIM tenure."

        elif evidence_key == "location_history":
            home = tx.customer_home_state or "Andhra Pradesh"
            loc = tx.location_city or ("Hyderabad" if is_hyderabad_scenario else "Local Area")
            t_str = tx.time_of_day or "02:13 AM"
            findings = {
                "registered_home_region": home,
                "transaction_location": loc,
                "transaction_time": t_str,
                "distance_km": round(tx.distance_from_home * 1.6, 1),
            }
            if is_hyderabad_scenario or tx.distance_from_home > 50:
                p_after = min(0.99, p_current + 0.08)
                u_after = max(0.05, u_current * 0.70)
                summary = f"Nighttime outlier ({t_str}) in {loc}, outside regular {home} zone."
            else:
                p_after = max(0.01, p_current - 0.15)
                u_after = max(0.05, u_current * 0.60)
                summary = f"Location matches cardholder expected radius in {home}."

        elif evidence_key == "merchant_history":
            m_name = tx.merchant_name or "New High-Risk Online Merchant"
            tx_count = tx.historical_merchant_tx_count if tx.historical_merchant_tx_count is not None else 0
            findings = {
                "merchant_name": m_name,
                "customer_past_transactions_at_merchant": tx_count,
                "mcc_category": tx.merchant_category,
                "merchant_risk_tier": "ELEVATED" if tx_count == 0 else "LOW",
            }
            if tx_count == 0:
                p_after = min(0.99, p_current + 0.09)
                u_after = max(0.05, u_current * 0.65)
                summary = f"First-time merchant engagement with elevated category chargeback score."
            else:
                p_after = max(0.01, p_current - 0.18)
                u_after = max(0.05, u_current * 0.50)
                summary = f"Established retailer relationship ({tx_count} prior successful payments)."

        elif evidence_key == "behavioral_stepup_2fa":
            success = tx.two_factor_auth_success
            if force_outcome == "PASS":
                success = 1
            elif force_outcome == "FAIL":
                success = 0
            elif success is None:
                success = 0 if (is_hyderabad_scenario or p_current > 0.70) else 1

            findings = {
                "channel": "SMS OTP & Authenticator App",
                "challenge_result": "SUCCESS" if success == 1 else "FAILED / TIMED OUT",
                "response_time_sec": 14 if success == 1 else 90,
            }
            if success == 1:
                p_after = max(0.02, p_current - 0.55)
                u_after = max(0.04, u_current * 0.30)
                summary = "Cryptographic 2FA step-up challenge verified successfully."
            else:
                p_after = min(0.99, p_current + 0.35)
                u_after = max(0.04, u_current * 0.30)
                summary = "2FA challenge timed out or failed authentication."
        else:
            p_after = p_current
            u_after = u_current
            findings = {}
            summary = "Unknown evidence type."

        return round(float(p_after), 4), round(float(u_after), 4), findings, summary

    def run_sequential_investigation(
        self,
        tx: Transaction,
        p_initial: float,
        u_initial: float,
        canonical_mode: Optional[str] = None,
    ) -> Tuple[float, float, List[InvestigationStep], str, bool]:
        """
        Executes the full sequential evidence acquisition process.
        """
        trajectory: List[InvestigationStep] = []
        p_curr = p_initial
        u_curr = u_initial
        cumulative_cost = 0.0

        if canonical_mode == "inconclusive_human":
            simulated_chain = [
                ("transaction_history", 0.70, 0.65, "Transaction history partially inconclusive; spend profile ambiguously aligned with split merchant."),
                ("device_history", 0.68, 0.62, "Device telemetry inconclusive: secondary tablet used 3 months ago."),
                ("location_history", 0.73, 0.60, "Location context boundary: near regional border, geo-velocity ambiguous."),
            ]
            for step_idx, (e_key, p_step, u_step, desc) in enumerate(simulated_chain, start=1):
                source = AVAILABLE_EVIDENCE_SOURCES[e_key]
                step = InvestigationStep(
                    step_number=step_idx,
                    evidence_type=e_key,
                    evidence_name=source.name,
                    cost=source.cost,
                    latency_ms=source.latency_ms,
                    p_fraud_before=round(p_curr, 4),
                    p_fraud_after=round(p_step, 4),
                    uncertainty_before=round(u_curr, 4),
                    uncertainty_after=round(u_step, 4),
                    uncertainty_reduction=round(max(0.0, u_curr - u_step), 4),
                    findings={"ambiguity_flag": True, "detail": desc},
                    summary=desc,
                )
                trajectory.append(step)
                p_curr = p_step
                u_curr = u_step
                cumulative_cost += source.cost

            reason = (
                f"Escalated to Human: Acquired {len(trajectory)} evidence sources (); "
                f"uncertainty remained high ({u_curr:.2f} > {self.tau_stop})."
            )
            return p_curr, u_curr, trajectory, reason, False

        unacquired = list(AVAILABLE_EVIDENCE_SOURCES.keys())
        max_steps = 3
        if canonical_mode == "single_step":
            max_steps = 1
        elif canonical_mode == "two_step":
            max_steps = 2

        step_counter = 1
        while unacquired and step_counter <= max_steps:
            ranked = self.rank_evidence(unacquired, u_curr)
            if not ranked:
                break

            best_key, best_voi = ranked[0]
            source = AVAILABLE_EVIDENCE_SOURCES[best_key]

            if cumulative_cost + source.cost > self.max_budget:
                reason = f"Budget exhausted: Next inquiry would exceed  threshold."
                return p_curr, u_curr, trajectory, reason, False

            p_next, u_next, findings, summary = self.simulate_query_evidence(
                best_key, tx, p_curr, u_curr
            )
            delta_u = max(0.0, u_curr - u_next)

            step = InvestigationStep(
                step_number=step_counter,
                evidence_type=best_key,
                evidence_name=source.name,
                cost=source.cost,
                latency_ms=source.latency_ms,
                p_fraud_before=round(p_curr, 4),
                p_fraud_after=round(p_next, 4),
                uncertainty_before=round(u_curr, 4),
                uncertainty_after=round(u_next, 4),
                uncertainty_reduction=round(delta_u, 4),
                findings=findings,
                summary=summary,
            )
            trajectory.append(step)

            p_curr = p_next
            u_curr = u_next
            cumulative_cost += source.cost
            unacquired.remove(best_key)
            step_counter += 1

            if u_curr <= self.tau_stop:
                if p_curr >= 0.80:
                    action_summary = "Flag/Block transaction autonomously"
                elif p_curr <= 0.30:
                    action_summary = "Approve transaction autonomously"
                else:
                    action_summary = "Resolved to low-risk threshold"
                reason = (
                    f"Stopping Rule Satisfied: Uncertainty collapsed to {u_curr:.2f} (<= {self.tau_stop}). "
                    f"Action: {action_summary}."
                )
                return p_curr, u_curr, trajectory, reason, True

            if delta_u < self.min_gain and step_counter > 2:
                reason = f"Diminishing Returns: Uncertainty reduction stalled (ΔU = {delta_u:.4f} < {self.min_gain})."
                return p_curr, u_curr, trajectory, reason, False

        reason = (
            f"Escalated to Human: Evaluated {len(trajectory)} evidence items (); "
            f"uncertainty remains inconclusive ({u_curr:.2f})."
        )
        return p_curr, u_curr, trajectory, reason, False
