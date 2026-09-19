"""
Generates investigator case briefings, explainability summaries, and verification checklists.
"""

from typing import Dict, Any, List
from safe_escalate.data.schema import DecisionPacket, Transaction


class InvestigatorExplainer:
    """
    Synthesizes machine learning outputs, conformal intervals, and feature attributions
    into an actionable human investigation dossier.
    """

    @staticmethod
    def generate_case_brief(packet: DecisionPacket, tx: Transaction) -> Dict[str, Any]:
        """
        Creates a structured dossier for the fraud analyst UI.
        """
        # Identify top anomaly drivers
        top_anomalies = []
        for feat, score in list(packet.feature_attributions.items())[:3]:
            if score > 15.0:
                top_anomalies.append(f"{feat.replace('_', ' ').title()} ({score}% impact)")

        anomaly_text = (
            ", ".join(top_anomalies)
            if top_anomalies
            else "General multi-feature borderline pattern"
        )

        # Actionable checklist
        checklist = []
        if tx.online_order == 1:
            checklist.append("Verify IP geolocation against cardholder billing address.")
        if tx.distance_from_home > 50:
            checklist.append(f"Cardholder is {tx.distance_from_home} miles away from billing residence.")
        if packet.evidence_collected and packet.evidence_collected.get("two_factor_auth_success") == 0:
            checklist.append("Urgent: 2FA challenge timed out or was rejected by cardholder.")
        if tx.amount > 1000:
            checklist.append(f"High exposure amount (${tx.amount:.2f}): Requires manager sign-off if approving.")
        checklist.append("Check recent card activity within last 24 hours for velocity bursts.")

        return {
            "transaction_id": packet.transaction_id,
            "amount": packet.amount,
            "risk_score_pct": round(packet.p_fraud_final * 100, 1),
            "conformal_set": packet.conformal_set,
            "uncertainty_profile": {
                "aleatoric": packet.aleatoric_uncertainty,
                "epistemic": packet.epistemic_uncertainty,
                "type": (
                    "Out-of-Distribution Anomaly"
                    if packet.epistemic_uncertainty > packet.aleatoric_uncertainty
                    else "Boundary Ambiguity"
                ),
            },
            "primary_drivers": anomaly_text,
            "investigator_checklist": checklist,
            "rationale": packet.investigator_rationale,
            "top_features": list(packet.feature_attributions.items())[:5],
        }
