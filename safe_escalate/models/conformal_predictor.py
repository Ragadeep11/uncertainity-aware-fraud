"""
Split Conformal Predictor for distribution-free finite-sample error control.
Guarantees marginal coverage P(Y in Gamma_alpha(X)) >= 1 - alpha.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from safe_escalate.models.base_classifier import BaseFraudClassifier


class ConformalFraudPredictor:
    """
    Inductive Split Conformal Prediction implementation.
    Transforms point-prediction probabilities into statistically valid prediction sets.
    """

    def __init__(self, base_classifier: BaseFraudClassifier, alpha: float = 0.05):
        self.base_classifier = base_classifier
        self.alpha = alpha
        self.q_hat: float = 1.0
        self.n_cal: int = 0
        self.cal_scores: np.ndarray = np.array([])
        self.is_calibrated: bool = False

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray):
        """
        Calibrates the non-conformity threshold on an independent holdout calibration set.
        Score function: s_i = 1 - P(Y = y_i | x_i).
        """
        probs = self.base_classifier.predict_proba(X_cal)
        self.n_cal = len(y_cal)

        # Compute non-conformity score for the true label: 1 - P(Y_true | x)
        true_class_probs = probs[np.arange(self.n_cal), y_cal]
        self.cal_scores = 1.0 - true_class_probs

        # Finite-sample adjusted quantile level
        q_level = np.ceil((self.n_cal + 1) * (1.0 - self.alpha)) / self.n_cal
        q_level = float(np.clip(q_level, 0.0, 1.0))

        # Quantile with method='higher' to preserve conservative finite-sample coverage guarantee
        self.q_hat = float(np.quantile(self.cal_scores, q_level, method="higher"))
        self.is_calibrated = True

    def predict_set(self, x_vector: np.ndarray) -> List[str]:
        """
        Returns the conformal prediction set for a single sample x.
        Labels: 'Legit' (0) and/or 'Fraud' (1).
        """
        if not self.is_calibrated:
            raise RuntimeError("Conformal predictor must be calibrated with a holdout set.")

        probs = self.base_classifier.predict_proba(x_vector.reshape(1, -1))[0]
        p_legit, p_fraud = probs[0], probs[1]

        prediction_set = []
        # Check class 0 (Legit)
        if (1.0 - p_legit) <= self.q_hat:
            prediction_set.append("Legit")
        # Check class 1 (Fraud)
        if (1.0 - p_fraud) <= self.q_hat:
            prediction_set.append("Fraud")

        # In conformal prediction with s = 1 - P(y|x), if neither class achieves 1 - q_hat,
        # the model is uncertain and cannot statistically exclude either class.
        # Thus, an empty set signifies maximum ambiguity:
        if not prediction_set:
            prediction_set = ["Legit", "Fraud"]

        return prediction_set

    def predict_sets_batch(self, X: np.ndarray) -> List[List[str]]:
        """Batch generation of prediction sets."""
        probs = self.base_classifier.predict_proba(X)
        sets = []
        for i in range(len(X)):
            p_legit, p_fraud = probs[i, 0], probs[i, 1]
            s = []
            if (1.0 - p_legit) <= self.q_hat:
                s.append("Legit")
            if (1.0 - p_fraud) <= self.q_hat:
                s.append("Fraud")
            if not s:
                s = ["Legit", "Fraud"]
            sets.append(s)
        return sets

    def evaluate_coverage(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluates empirical coverage, average set size, and ambiguity rate.
        """
        pred_sets = self.predict_sets_batch(X_test)
        covered = 0
        ambiguous = 0
        singleton_legit = 0
        singleton_fraud = 0

        label_map = {0: "Legit", 1: "Fraud"}

        for i, pset in enumerate(pred_sets):
            true_label_str = label_map[y_test[i]]
            if true_label_str in pset:
                covered += 1
            if len(pset) > 1:
                ambiguous += 1
            elif pset == ["Legit"]:
                singleton_legit += 1
            elif pset == ["Fraud"]:
                singleton_fraud += 1

        n = len(y_test)
        return {
            "target_coverage": 1.0 - self.alpha,
            "empirical_coverage": covered / n,
            "ambiguity_rate": ambiguous / n,
            "singleton_legit_rate": singleton_legit / n,
            "singleton_fraud_rate": singleton_fraud / n,
            "q_hat": self.q_hat,
        }
