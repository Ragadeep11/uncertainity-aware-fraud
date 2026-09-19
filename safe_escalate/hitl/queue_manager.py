"""
Priority queue manager for Human-in-the-Loop (HITL) fraud investigation triage.
"""

from typing import List, Dict, Any, Optional
import time
from safe_escalate.data.schema import DecisionPacket, Transaction


class HITLQueueManager:
    """
    Manages pending human review cases, prioritizing by financial risk exposure and uncertainty severity.
    """

    def __init__(self):
        # Maps transaction_id -> Dict containing packet, tx, priority, timestamp
        self._queue: Dict[str, Dict[str, Any]] = {}
        self._history: List[Dict[str, Any]] = []

    def enqueue(self, packet: DecisionPacket, tx: Transaction) -> float:
        """
        Adds an escalated transaction to the triage queue with a computed priority score.
        Priority = Amount * (0.6 * p_fraud_final + 0.4 * epistemic_uncertainty).
        """
        risk_weight = 0.6 * packet.p_fraud_final + 0.4 * packet.epistemic_uncertainty
        priority = round(float(packet.amount * max(0.1, risk_weight)), 2)

        entry = {
            "transaction_id": packet.transaction_id,
            "packet": packet.model_dump(),
            "transaction": tx.model_dump(),
            "priority": priority,
            "enqueued_at": time.time(),
            "status": "PENDING",
        }
        self._queue[packet.transaction_id] = entry
        return priority

    def get_pending(self) -> List[Dict[str, Any]]:
        """Returns all pending cases sorted by priority descending (highest risk first)."""
        pending = list(self._queue.values())
        return sorted(pending, key=lambda x: x["priority"], reverse=True)

    def resolve(
        self, transaction_id: str, decision: str, analyst_id: str = "analyst_01", notes: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Records human analyst judgment ('APPROVE' or 'DECLINE').
        """
        if transaction_id not in self._queue:
            return None

        entry = self._queue.pop(transaction_id)
        entry["status"] = "RESOLVED"
        entry["human_decision"] = decision
        entry["analyst_id"] = analyst_id
        entry["analyst_notes"] = notes
        entry["resolved_at"] = time.time()

        self._history.append(entry)
        return entry

    def get_queue_stats(self) -> Dict[str, Any]:
        """Summary metrics for the investigator dashboard."""
        pending_count = len(self._queue)
        resolved_count = len(self._history)
        total_value = sum(item["packet"]["amount"] for item in self._queue.values())

        approved_count = sum(1 for h in self._history if h["human_decision"] == "APPROVE")
        declined_count = sum(1 for h in self._history if h["human_decision"] == "DECLINE")

        return {
            "pending_count": pending_count,
            "resolved_count": resolved_count,
            "pending_exposure_usd": round(total_value, 2),
            "analyst_approvals": approved_count,
            "analyst_declines": declined_count,
        }

    def clear(self):
        """Clears queue and history."""
        self._queue.clear()
        self._history.clear()
