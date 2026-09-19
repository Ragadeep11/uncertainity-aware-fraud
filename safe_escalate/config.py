"""
System configuration, cost matrix specifications, and uncertainty hyperparameters.
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class CostConfig:
    """
    Economic parameters for evaluating decision and escalation policies.
    """
    # Customer friction / churn cost for falsely declining a legitimate transaction
    cost_false_positive: float = 20.0

    # Chargeback penalty multiplier added to transaction amount when fraud is missed
    cost_false_negative_penalty: float = 15.0

    # Direct API/telecom fee per Tier-2 dynamic evidence query (e.g. 2FA SMS / carrier lookup)
    cost_evidence_acquisition: float = 0.20

    # Fully loaded labor cost per human investigator review
    cost_human_review: float = 12.0

    # Customer friction cost for prompting 2FA when legit (minor inconvenience)
    cost_2fa_friction: float = 1.0


@dataclass
class UncertaintyConfig:
    """
    Hyperparameters for Conformal Prediction and Epistemic/Aleatoric estimation.
    """
    # Conformal prediction significance level (alpha = 0.05 => 95% coverage guarantee)
    alpha: float = 0.05

    # Aleatoric threshold: normalized entropy > threshold implies severe decision boundary ambiguity
    aleatoric_threshold: float = 0.65

    # Epistemic threshold: distance/OOD metric percentile to flag unfamiliar transaction patterns
    epistemic_threshold: float = 0.70

    # High-value transaction threshold where model uncertainty strictly mandates human oversight
    high_value_amount_threshold: float = 1000.0


@dataclass
class AppConfig:
    """
    Top-level application settings.
    """
    costs: CostConfig = field(default_factory=CostConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    random_seed: int = 42
    dataset_size: int = 15000
    test_split: float = 0.2
    calibration_split: float = 0.2  # Fraction of training set used for conformal calibration

    def to_dict(self) -> Dict[str, Any]:
        return {
            "costs": self.costs.__dict__,
            "uncertainty": self.uncertainty.__dict__,
            "random_seed": self.random_seed,
            "dataset_size": self.dataset_size,
        }
