"""
Cost-utility matrix and expected loss calculations for fraud decision policies.
"""

from safe_escalate.config import CostConfig


class CostMatrixEvaluator:
    """
    Computes economic loss and operational cost for individual decisions and aggregated batches.
    """

    def __init__(self, cost_config: CostConfig = None):
        self.config = cost_config or CostConfig()

    def evaluate_realized_cost(
        self, action: str, is_fraud: int, amount: float, tier: int
    ) -> float:
        """
        Calculates ground-truth realized operational cost for a processed transaction.
        """
        cost = 0.0

        # Tier query / labor overheads
        if tier == 1:
            cost += self.config.cost_evidence_acquisition
        elif tier == 2:
            cost += (self.config.cost_evidence_acquisition + self.config.cost_human_review)

        # Classification outcome costs
        if action in ("APPROVE", "STEP_UP_RESOLVED") and action.startswith("APPROVE") or (action == "STEP_UP_RESOLVED" and is_fraud == 0):
            # If approved:
            if is_fraud == 1:
                # False Negative: direct fraud loss + chargeback penalty
                cost += (amount + self.config.cost_false_negative_penalty)
            else:
                # True Negative: $0 (clean execution)
                pass
        elif action == "DECLINE" or (action == "STEP_UP_RESOLVED" and is_fraud == 1):
            if is_fraud == 0:
                # False Positive: customer friction / churn cost
                cost += self.config.cost_false_positive
            else:
                # True Positive: successfully stopped fraud ($0 fraud loss)
                pass
        elif action == "HUMAN_ESCALATION":
            # Human reviews achieve high accuracy (assumed 97% human accuracy)
            # Remaining 3% error rate
            if is_fraud == 1:
                # 3% chance human mislabels fraud
                cost += 0.03 * (amount + self.config.cost_false_negative_penalty)
            else:
                # 2% chance human mislabels legit
                cost += 0.02 * self.config.cost_false_positive

        return round(float(cost), 2)
