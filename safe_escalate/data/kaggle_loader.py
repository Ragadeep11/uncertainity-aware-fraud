"""
Kaggle Credit Card Fraud Dataset (ULB / European Cardholders) Loader.
Pure Python standard library + NumPy (Zero pandas dependency).
Handles caching, automated download from verified mirrors, stratified splitting,
and mapping into SafeEscalate domain models.
"""

import os
import csv
import zipfile
import urllib.request
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from safe_escalate.data.schema import Transaction

DATASET_ZIP_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/creditcard.csv.zip"
DATASET_CSV_URL = "https://huggingface.co/datasets/JEFFREY-VERDIERE/Creditcard/resolve/main/creditcard.csv"

DATA_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = DATA_DIR / "creditcard.csv"
DEFAULT_ZIP_PATH = DATA_DIR / "creditcard.csv.zip"

CANDIDATE_PATHS = [
    DEFAULT_CSV_PATH,
    Path.home() / "Downloads" / "creditcard.csv",
    Path("creditcard.csv").resolve(),
    Path("data") / "creditcard.csv",
]

ZIP_CANDIDATES = [
    DEFAULT_ZIP_PATH,
    DATA_DIR / "creditcard.zip",
    Path.home() / "Downloads" / "creditcard.csv.zip",
    Path.home() / "Downloads" / "creditcard.zip",
    Path.home() / "Downloads" / "archive.zip",
    Path("creditcard.csv.zip").resolve(),
]


class KaggleDatasetLoader:
    """
    Manages loading, caching, stratified splitting, and feature extraction
    for the real-world Kaggle Credit Card Fraud Detection dataset (284,807 transactions).
    High-performance parser using pure Python and NumPy with zero pandas dependency.
    """

    FEATURE_NAMES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    SECONDARY_FEATURE_NAMES = [
        "device_trust_score",
        "carrier_sim_swap_age_days",
        "ip_country_match",
        "two_factor_auth_success",
    ]

    def __init__(self, csv_path: Optional[str] = None, random_seed: int = 42):
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.csv_path = Path(csv_path) if csv_path else self._locate_or_download()

    @classmethod
    def _extract_zip(cls, zip_file: Path, target_csv: Path) -> Path:
        print(f"[KaggleLoader] Extracting archive {zip_file} to {target_csv}...")
        with zipfile.ZipFile(zip_file, "r") as zf:
            for item in zf.namelist():
                if item.endswith(".csv"):
                    with zf.open(item) as src, open(target_csv, "wb") as dst:
                        chunk_size = 1024 * 1024
                        while True:
                            chunk = src.read(chunk_size)
                            if not chunk:
                                break
                            dst.write(chunk)
                    print(f"[KaggleLoader] Successfully extracted {target_csv} ({target_csv.stat().st_size / (1024*1024):.1f} MB)")
                    return target_csv
        raise ValueError(f"No CSV file found inside archive {zip_file}")

    @classmethod
    def _locate_or_download(cls) -> Path:
        """Finds existing creditcard.csv or downloads it into safe_escalate/data/creditcard.csv."""
        # 1. Check if uncompressed CSV exists
        for candidate in CANDIDATE_PATHS:
            if candidate.exists() and candidate.stat().st_size > 10_000_000:
                print(f"[KaggleLoader] Using existing local dataset at: {candidate}")
                return candidate

        # 2. Check if compressed zip archive exists
        for z_path in ZIP_CANDIDATES:
            if z_path.exists() and z_path.stat().st_size > 10_000_000:
                return cls._extract_zip(z_path, DEFAULT_CSV_PATH)

        DEFAULT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"[KaggleLoader] Dataset not found locally. Streaming zip archive from mirror: {DATASET_ZIP_URL}")
        print(f"[KaggleLoader] Target destination: {DEFAULT_ZIP_PATH}")

        temp_target = DEFAULT_CSV_PATH.with_suffix(".tmp")
        req = urllib.request.Request(
            DATASET_URL,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )

        try:
            with urllib.request.urlopen(req, timeout=180) as resp, open(temp_target, "wb") as out_file:
                total_size = int(resp.headers.get("Content-Length", 150_000_000))
                downloaded = 0
                chunk_size = 1024 * 512  # 512 KB chunks
                last_pct = -1

                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    pct = int(downloaded * 100 / total_size) if total_size else 0
                    if pct != last_pct and pct % 10 == 0:
                        print(f"  Downloaded {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({pct}%)...")
                        last_pct = pct

            temp_target.rename(DEFAULT_CSV_PATH)
            print(f"[KaggleLoader] Successfully downloaded creditcard.csv ({DEFAULT_CSV_PATH.stat().st_size / (1024*1024):.1f} MB)")
            return DEFAULT_CSV_PATH

        except Exception as e:
            if temp_target.exists():
                temp_target.unlink()
            raise RuntimeError(
                f"Failed to automatically download Kaggle credit card dataset: {e}. "
                f"Please manually download creditcard.csv from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
                f"and place it in {DEFAULT_CSV_PATH} or {Path.home() / 'Downloads' / 'creditcard.csv'}."
            )

    def load_raw_data(self, max_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parses CSV directly into NumPy arrays. If max_samples is provided, retains
        ALL 492 fraud cases and samples (max_samples - 492) legitimate cases.
        """
        print(f"[KaggleLoader] Fast-streaming CSV: {self.csv_path}...")
        fraud_rows = []
        legit_rows = []

        with open(self.csv_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            first_row = next(reader)
            clean_first = [c.strip().strip('"').lower() for c in first_row]

            has_header = any(keyword in clean_first for keyword in ["time", "v1", "class", "amount"])

            if has_header:
                col_map = {col: idx for idx, col in enumerate(clean_first)}
                feat_indices = [col_map[col.lower()] for col in self.FEATURE_NAMES if col.lower() in col_map]
                class_idx = col_map.get("class", len(clean_first) - 1)
            else:
                # Direct index mapping: 0=Time, 1..28=V1..V28, 29=Amount, 30=Class
                feat_indices = list(range(30))
                class_idx = 30
                # Process first_row since it's already a valid data record
                try:
                    cls_val = int(float(first_row[class_idx]))
                    feat_vals = [float(first_row[idx]) for idx in feat_indices]
                    if cls_val == 1:
                        fraud_rows.append(feat_vals)
                    else:
                        legit_rows.append(feat_vals)
                except (ValueError, IndexError):
                    pass

            for row in reader:
                if not row or len(row) <= class_idx:
                    continue
                try:
                    cls_val = int(float(row[class_idx]))
                    feat_vals = [float(row[idx]) for idx in feat_indices]
                    if cls_val == 1:
                        fraud_rows.append(feat_vals)
                    else:
                        legit_rows.append(feat_vals)
                except (ValueError, IndexError):
                    continue

        n_fraud = len(fraud_rows)
        n_legit = len(legit_rows)
        n_total = n_fraud + n_legit
        print(f"[KaggleLoader] Parsed records: {n_total:,} | Fraud cases: {n_fraud} ({n_fraud/n_total*100:.3f}%) | Legit: {n_legit:,}")

        # Stratified sampling if requested
        if max_samples is not None and max_samples < n_total:
            n_target_fraud = min(n_fraud, max_samples // 4) if max_samples < 2000 else n_fraud
            n_target_legit = max_samples - n_target_fraud

            if n_target_legit < len(legit_rows):
                legit_arr = np.array(legit_rows, dtype=np.float64)
                sub_idx = self.rng.choice(len(legit_arr), size=n_target_legit, replace=False)
                legit_rows = legit_arr[sub_idx].tolist()

        X_f = np.array(fraud_rows, dtype=np.float64)
        X_l = np.array(legit_rows, dtype=np.float64)
        y_f = np.ones(len(fraud_rows), dtype=int)
        y_l = np.zeros(len(legit_rows), dtype=int)

        X = np.vstack([X_f, X_l])
        y = np.concatenate([y_f, y_l])

        perm = self.rng.permutation(len(y))
        return X[perm], y[perm]

    def generate_splits(
        self,
        max_samples: Optional[int] = 50000,
        train_ratio: float = 0.60,
        cal_ratio: float = 0.20,
    ) -> Dict[str, Any]:
        """
        Creates stratified Train (60%), Conformal Calibration (20%), and Test Stream (20%) splits.
        Synthesizes conditioned Tier-2 telemetry for active step-up simulation.
        """
        X, y = self.load_raw_data(max_samples=max_samples)

        # Stratified splitting indices
        fraud_indices = np.where(y == 1)[0]
        legit_indices = np.where(y == 0)[0]

        self.rng.shuffle(fraud_indices)
        self.rng.shuffle(legit_indices)

        # Split fraud
        n_f_train = int(len(fraud_indices) * train_ratio)
        n_f_cal = int(len(fraud_indices) * cal_ratio)
        f_train_idx = fraud_indices[:n_f_train]
        f_cal_idx = fraud_indices[n_f_train : n_f_train + n_f_cal]
        f_test_idx = fraud_indices[n_f_train + n_f_cal :]

        # Split legit
        n_l_train = int(len(legit_indices) * train_ratio)
        n_l_cal = int(len(legit_indices) * cal_ratio)
        l_train_idx = legit_indices[:n_l_train]
        l_cal_idx = legit_indices[n_l_train : n_l_train + n_l_cal]
        l_test_idx = legit_indices[n_l_train + n_l_cal :]

        train_idx = np.concatenate([f_train_idx, l_train_idx])
        cal_idx = np.concatenate([f_cal_idx, l_cal_idx])
        test_idx = np.concatenate([f_test_idx, l_test_idx])

        self.rng.shuffle(train_idx)
        self.rng.shuffle(cal_idx)
        self.rng.shuffle(test_idx)

        X_train, y_train = X[train_idx], y[train_idx]
        X_cal, y_cal = X[cal_idx], y[cal_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        print(f"[KaggleLoader] Splits prepared: Train={len(y_train):,} ({y_train.sum()} fraud) | "
              f"Calibration={len(y_cal):,} ({y_cal.sum()} fraud) | "
              f"Test={len(y_test):,} ({y_test.sum()} fraud)")

        # Generate conditioned Tier 2 telemetry for full train set
        tier2_train = self._generate_conditioned_telemetry(y_train, X_train)
        X_train_full = np.hstack([X_train, tier2_train])

        # Generate Transaction domain objects for the test stream
        test_txs = self._create_transaction_objects(X_test, y_test)

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_train_full": X_train_full,
            "X_cal": X_cal,
            "y_cal": y_cal,
            "X_test": X_test,
            "y_test": y_test,
            "test_transactions": test_txs,
            "feature_names": self.FEATURE_NAMES,
            "n_features": X.shape[1],
        }

    def _generate_conditioned_telemetry(self, y: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Synthesizes realistic Tier-2 active verification signals conditioned on ground truth."""
        n = len(y)
        device_trust = np.where(
            y == 0,
            np.clip(self.rng.beta(8, 2, size=n), 0.35, 1.0),
            np.clip(self.rng.beta(2, 6, size=n), 0.05, 0.65),
        )
        sim_swap_age = np.where(
            y == 0,
            self.rng.integers(120, 1500, size=n),
            np.clip(self.rng.exponential(scale=10.0, size=n), 0, 90).astype(int),
        )
        ip_match = np.where(
            y == 0,
            self.rng.binomial(1, 0.95, size=n),
            self.rng.binomial(1, 0.20, size=n),
        )
        two_fa = np.where(
            y == 0,
            self.rng.binomial(1, 0.97, size=n),
            self.rng.binomial(1, 0.08, size=n),
        )

        return np.column_stack([device_trust, sim_swap_age, ip_match, two_fa])

    def _create_transaction_objects(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> List[Transaction]:
        """Maps numpy matrix rows into Pydantic Transaction models."""
        n = len(y_test)
        sec_telemetry = self._generate_conditioned_telemetry(y_test, X_test)

        # In FEATURE_NAMES: [Time, V1..V28, Amount]
        time_idx = 0
        amount_idx = 29

        transactions: List[Transaction] = []
        for i in range(n):
            amt = float(X_test[i, amount_idx])
            if amt <= 0:
                amt = 0.01

            pca_dict = {"Time": round(float(X_test[i, time_idx]), 1)}
            for j in range(1, 29):
                pca_dict[f"V{j}"] = round(float(X_test[i, j]), 4)
            pca_dict["Amount"] = round(amt, 2)

            v1_val = float(X_test[i, 1])
            v2_val = float(X_test[i, 2])
            v3_val = float(X_test[i, 3])
            v4_val = float(X_test[i, 4])
            v5_val = float(X_test[i, 5])
            v6_val = float(X_test[i, 6])
            v7_val = float(X_test[i, 7])

            tx = Transaction(
                transaction_id=f"KAGGLE-TX-{100000 + i}",
                amount=round(amt, 2),
                merchant_category=int(abs(v1_val) % 10),
                distance_from_home=round(abs(v1_val) * 12.0 + 3.0, 1),
                distance_from_last_tx=round(abs(v2_val) * 6.0 + 1.0, 1),
                ratio_to_median_price=round(max(0.1, amt / 88.0), 2),
                repeat_retailer=1 if v3_val > -1.0 else 0,
                used_chip=1 if v4_val < 1.5 else 0,
                used_pin=0,
                online_order=1 if abs(v5_val) > 1.2 else 0,
                velocity_1h=max(0, int(abs(v6_val) * 2)),
                velocity_24h=max(1, int(abs(v7_val) * 4 + 2)),
                device_trust_score=round(float(sec_telemetry[i, 0]), 3),
                carrier_sim_swap_age_days=int(sec_telemetry[i, 1]),
                ip_country_match=int(sec_telemetry[i, 2]),
                two_factor_auth_success=int(sec_telemetry[i, 3]),
                is_fraud=int(y_test[i]),
                pca_features=pca_dict,
            )
            transactions.append(tx)

        return transactions
