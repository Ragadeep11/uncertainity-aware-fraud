"""
Data schemas for transactions, evidence payloads, and decision outputs.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict


class Transaction(BaseModel):
    """
    Financial transaction feature representation.
    Supports both tabular features and Kaggle PCA features (V1..V28, Time).
    """
    model_config = ConfigDict(extra="allow")

    transaction_id: str
    amount: float = Field(..., gt=0)
    merchant_category: int = Field(default=1, ge=0, le=10)
    distance_from_home: float = Field(default=5.0, ge=0)
    distance_from_last_tx: float = Field(default=1.0, ge=0)
    ratio_to_median_price: float = Field(default=1.0, ge=0)
    repeat_retailer: int = Field(default=1, ge=0, le=1)
    used_chip: int = Field(default=1, ge=0, le=1)
    used_pin: int = Field(default=0, ge=0, le=1)
    online_order: int = Field(default=0, ge=0, le=1)
    velocity_1h: int = Field(default=1, ge=0)
    velocity_24h: int = Field(default=2, ge=0)

    # Kaggle PCA feature dictionary (V1..V28, Time)
    pca_features: Optional[Dict[str, float]] = None

    # Tier-2 Dynamic Evidence Features (retrieved on-demand)
    device_trust_score: Optional[float] = None
    carrier_sim_swap_age_days: Optional[int] = None
    ip_country_match: Optional[int] = None
    two_factor_auth_success: Optional[int] = None

    # Ground truth (if available for testing/simulation)
    is_fraud: Optional[int] = None


class EvidencePayload(BaseModel):
    """
    Dynamically queried micro-evidence returned during Tier-2 escalation.
    """
    device_trust_score: float
    carrier_sim_swap_age_days: int
    ip_country_match: int
    two_factor_auth_success: int
    evidence_cost: float
    query_latency_ms: int


class DecisionPacket(BaseModel):
    """
    Complete audit trail and decision breakdown for a transaction.
    """
    transaction_id: str
    amount: float
    p_fraud_initial: float
    p_fraud_final: float
    conformal_set: List[str]
    is_ambiguous: bool
    aleatoric_uncertainty: float
    epistemic_uncertainty: float
    final_action: str  # "APPROVE", "DECLINE", "STEP_UP_RESOLVED", "HUMAN_ESCALATION"
    escalation_tier: int  # 0, 1, 2
    evidence_collected: Optional[Dict[str, Any]] = None
    investigator_rationale: Optional[str] = None
    feature_attributions: Dict[str, float] = Field(default_factory=dict)
    ground_truth: Optional[int] = None
    operational_cost: float = 0.0
