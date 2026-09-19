"""
Uncertainty estimation and decomposition into Aleatoric and Epistemic components,
plus feature attribution for human investigator interpretability.
"""

import numpy as np
from typing import Dict, Tuple, Any
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class UncertaintyEstimator:
    """
    Decomposes model prediction uncertainty into:
    1. Aleatoric Uncertainty: Intrinsic ambiguity / boundary noise via normalized entropy.
    2. Epistemic Uncertainty: Lack of model knowledge / Out-Of-Distribution (OOD) distance.
    """

    def __init__(self, n_neighbors: int = 15):
        self.n_neighbors = n_neighbors
        self.scaler = StandardScaler()
        self.nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", n_jobs=-1)
        self.train_dist_95: float = 1.0
        self.feature_means: np.ndarray = np.array([])
        self.feature_stds: np.ndarray = np.array([])
        self.is_fitted: bool = False

    def fit(self, X_train: np.ndarray):
        """Fit scaler and k-NN reference space for epistemic OOD evaluation."""
        X_norm = self.scaler.fit_transform(X_train)

        # For high-throughput scalability (e.g. 50k-284k rows in Kaggle),
        # use an anchor reference set of up to 4,000 points for k-NN manifold distance.
        if len(X_norm) > 4000:
            rng = np.random.default_rng(42)
            anchor_idx = rng.choice(len(X_norm), size=4000, replace=False)
            X_anchor = X_norm[anchor_idx]
        else:
            X_anchor = X_norm

        self.nn_model.fit(X_anchor)

        # Baseline distance statistics for calibration (evaluated on up to 2,000 points)
        eval_sample = X_anchor[:min(2000, len(X_anchor))]
        distances, _ = self.nn_model.kneighbors(eval_sample)
        mean_knn_dists = np.mean(distances, axis=1)
        self.train_dist_95 = float(np.percentile(mean_knn_dists, 95))
        if self.train_dist_95 <= 1e-6:
            self.train_dist_95 = 1.0

        self.feature_means = np.mean(X_train, axis=0)
        self.feature_stds = np.std(X_train, axis=0) + 1e-6
        self.is_fitted = True

    def calculate_aleatoric(self, p_fraud: float) -> float:
        """
        Normalized binary Shannon Entropy in [0.0, 1.0].
        U_a = - (p log2(p) + (1-p) log2(1-p)).
        Maximized at p = 0.5 (maximum ambiguity).
        """
        p = np.clip(p_fraud, 1e-7, 1.0 - 1e-7)
        entropy = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
        return float(np.clip(entropy, 0.0, 1.0))

    def calculate_epistemic(self, x_vector: np.ndarray) -> float:
        """
        Measures how far x is from known training clusters.
        Normalized distance to nearest neighbors in standardized feature space.
        """
        if not self.is_fitted:
            raise RuntimeError("UncertaintyEstimator must be fitted.")

        x_norm = self.scaler.transform(x_vector.reshape(1, -1))
        dists, _ = self.nn_model.kneighbors(x_norm)
        mean_dist = float(np.mean(dists))

        # Normalized to roughly [0, 1] via hyperbolic tangent scaling against 95th percentile
        epistemic_score = float(np.tanh(mean_dist / (self.train_dist_95 * 1.2)))
        return float(np.clip(epistemic_score, 0.0, 1.0))

    def decompose(self, x_vector: np.ndarray, p_fraud: float) -> Tuple[float, float, float]:
        """
        Returns (total_uncertainty, aleatoric_uncertainty, epistemic_uncertainty).
        Total is a harmonized composite score.
        """
        u_aleatoric = self.calculate_aleatoric(p_fraud)
        u_epistemic = self.calculate_epistemic(x_vector)

        # Composite uncertainty score combining boundary ambiguity and novelty
        u_total = float(np.clip(0.6 * u_aleatoric + 0.4 * u_epistemic, 0.0, 1.0))
        return u_total, u_aleatoric, u_epistemic

    def compute_feature_attributions(
        self, x_vector: np.ndarray, feature_names: list
    ) -> Dict[str, float]:
        """
        Computes standardized z-score deviation for each feature relative to training distribution.
        Highlights why a transaction looks anomalous to investigators.
        """
        if not self.is_fitted:
            return {}

        z_scores = np.abs((x_vector - self.feature_means) / self.feature_stds)
        total_z = float(np.sum(z_scores)) + 1e-6

        # Normalized relative percentage contribution
        attributions = {}
        for name, z in zip(feature_names, z_scores):
            attributions[name] = round(float(z / total_z * 100.0), 1)

        # Return sorted by importance
        return dict(sorted(attributions.items(), key=lambda item: item[1], reverse=True))
