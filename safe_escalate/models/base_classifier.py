"""
Calibrated probabilistic base risk model and secondary Tier-2 validation model.
"""

import numpy as np
from typing import Tuple
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler


class BaseFraudClassifier:
    """
    Tier 1 Base Classifier: Fast, low-latency probability estimator
    operating exclusively on standard real-time transaction features.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        # HistGradientBoostingClassifier is fast, handles non-linear interactions and tabular data well
        self.estimator = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_depth=6,
            random_state=self.random_state,
            class_weight="balanced",
        )
        self.calibrator = None
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fit scaler, base estimator, and probability calibration wrapper."""
        X_scaled = self.scaler.fit_transform(X_train)
        # Wrap with isotonic calibration for well-calibrated posterior probabilities
        self.calibrator = CalibratedClassifierCV(
            estimator=self.estimator, method="isotonic", cv=3
        )
        self.calibrator.fit(X_scaled, y_train)
        self.is_fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns [P(Legit), P(Fraud)] for inputs."""
        if not self.is_fitted:
            raise RuntimeError("Base classifier must be fitted before predict_proba.")
        X_scaled = self.scaler.transform(X)
        return self.calibrator.predict_proba(X_scaled)

    def predict_p_fraud(self, X: np.ndarray) -> np.ndarray:
        """Returns 1D array of P(Fraud | x)."""
        probs = self.predict_proba(X)
        return probs[:, 1]


class Tier2EvidenceValidator:
    """
    Tier 2 Validator: Enriched risk model trained on base features
    PLUS dynamic secondary evidence (device trust, carrier SIM swap, 2FA challenge).
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.estimator = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=self.random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
        self.is_fitted = False

    def fit(self, X_full: np.ndarray, y_train: np.ndarray):
        """Fit on full vector [Base + Secondary Features]."""
        X_scaled = self.scaler.fit_transform(X_full)
        self.estimator.fit(X_scaled, y_train)
        self.is_fitted = True

    def predict_proba(self, X_full: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Tier2Validator must be fitted before predict_proba.")
        X_scaled = self.scaler.transform(X_full)
        return self.estimator.predict_proba(X_scaled)

    def predict_p_fraud(self, X_full: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X_full)
        return probs[:, 1]
