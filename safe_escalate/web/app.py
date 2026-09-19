"""
FastAPI application serving REST endpoints and the interactive SafeEscalate dashboard.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import os
from pathlib import Path

from safe_escalate.config import AppConfig, CostConfig, UncertaintyConfig
from safe_escalate.data.schema import Transaction, DecisionPacket
from safe_escalate.data.dataset_generator import TransactionDatasetGenerator
from safe_escalate.eval.benchmark import BenchmarkSuite
from safe_escalate.hitl.queue_manager import HITLQueueManager
from safe_escalate.hitl.explainer import InvestigatorExplainer

# Directory paths
WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"
TEMPLATES_DIR = WEB_DIR / "templates"

app = FastAPI(
    title="SafeEscalate: Uncertainty-Aware Fraud Escalation Framework",
    description="Production & Research API for Conformal Uncertainty and Cascaded Human-in-the-Loop Triaging.",
    version="1.0.0",
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# In-memory application state
STATE: Dict[str, Any] = {
    "engine": None,
    "queue_manager": HITLQueueManager(),
    "config": AppConfig(),
    "test_txs": [],
    "benchmark_results": None,
    "tx_stream_idx": 0,
}


def initialize_system(n_samples: int = 50000, dataset_type: Optional[str] = None):
    """Pre-trains base models, conformal predictor, and validators on Kaggle or Synthetic data."""
    if dataset_type is None:
        dataset_type = os.environ.get("SAFE_ESCALATE_DATASET", "kaggle")

    print(f"Training SafeEscalate core models on dataset: '{dataset_type}' (samples: {n_samples})...")
    suite = BenchmarkSuite(config=STATE["config"], dataset_type=dataset_type)
    try:
        engine, test_txs, cal_metrics = suite.prepare_experiment(
            n_samples=n_samples, dataset_type=dataset_type
        )
        STATE["dataset_type"] = dataset_type
        STATE["dataset_name"] = "Kaggle CreditCard Fraud (ULB)" if dataset_type == "kaggle" else "Synthetic Multi-Tier"
    except Exception as e:
        print(f"[Warning] Failed to initialize '{dataset_type}': {e}. Falling back to 'synthetic' generator.")
        engine, test_txs, cal_metrics = suite.prepare_experiment(
            n_samples=10000, dataset_type="synthetic"
        )
        STATE["dataset_type"] = "synthetic"
        STATE["dataset_name"] = "Synthetic Multi-Tier (Fallback)"

    STATE["engine"] = engine
    STATE["test_txs"] = test_txs
    STATE["conformal_metrics"] = cal_metrics
    print(f"System successfully initialized with {len(test_txs)} test transactions ready for evaluation.")


@app.on_event("startup")
async def startup_event():
    dataset = os.environ.get("SAFE_ESCALATE_DATASET", "kaggle")
    initialize_system(n_samples=50000, dataset_type=dataset)


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = TEMPLATES_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>SafeEscalate Dashboard Template Not Found</h1>", status_code=404)
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/status")
async def get_status():
    if STATE["engine"] is None:
        return {"status": "INITIALIZING"}
    engine = STATE["engine"]
    return {
        "status": "ONLINE",
        "dataset_name": STATE.get("dataset_name", "Kaggle CreditCard Fraud (ULB)"),
        "dataset_type": STATE.get("dataset_type", "kaggle"),
        "test_stream_pool": len(STATE.get("test_txs", [])),
        "conformal_q_hat": engine.conformal_predictor.q_hat,
        "conformal_alpha": engine.conformal_predictor.alpha,
        "conformal_target_coverage": 1.0 - engine.conformal_predictor.alpha,
        "queue_stats": STATE["queue_manager"].get_queue_stats(),
        "config": STATE["config"].to_dict(),
    }


@app.post("/api/predict", response_model=DecisionPacket)
async def process_transaction(tx: Transaction, canonical_mode: Optional[str] = None):
    if STATE["engine"] is None:
        raise HTTPException(status_code=503, detail="System models not yet initialized.")

    engine = STATE["engine"]
    packet = engine.process_transaction(tx, canonical_mode=canonical_mode)

    # If escalated to human review, enqueue in HITL manager
    if packet.escalation_tier == 2:
        STATE["queue_manager"].enqueue(packet, tx)

    return packet


def parse_smart_int(val, default=0):
    if val is None:
        return default
    s = str(val).strip().lower()
    if s in ("", "none", "null", "nan"):
        return default
    if s in ("true", "yes", "y", "t", "1"):
        return 1
    if s in ("false", "no", "n", "f", "0"):
        return 0
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return default


def parse_smart_float(val, default=None):
    if val is None:
        return default
    s = str(val).strip().lower()
    if s in ("", "none", "null", "nan"):
        return default
    try:
        return float(s)
    except (ValueError, TypeError):
        return default


def parse_smart_opt_int(val):
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in ("", "none", "null", "nan"):
        return None
    if s in ("true", "yes", "y", "t", "1"):
        return 1
    if s in ("false", "no", "n", "f", "0"):
        return 0
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None


def parse_smart_opt_float(val):
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in ("", "none", "null", "nan"):
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


CSV_ALIAS_MAP = {
    "amount": ["amount", "amt", "transactionamt", "price", "tx_amount", "value"],
    "transaction_id": ["transaction_id", "tx_id", "id", "transactionid", "txid", "tx_no"],
    "merchant_category": ["merchant_category", "merchant", "category", "mcc", "merchantcategory"],
    "distance_from_home": ["distance_from_home", "distance", "dist_from_home", "dist_home", "disthome", "home_distance"],
    "distance_from_last_tx": ["distance_from_last_tx", "dist_from_last_tx", "dist_last", "distlast", "distance_last"],
    "ratio_to_median_price": ["ratio_to_median_price", "ratio", "ratiomedian", "ratio_median", "price_ratio"],
    "repeat_retailer": ["repeat_retailer", "repeat", "is_repeat", "repeatretailer"],
    "used_chip": ["used_chip", "chip", "is_chip", "usedchip", "emv"],
    "used_pin": ["used_pin", "pin", "is_pin", "usedpin"],
    "online_order": ["online_order", "online", "is_online", "onlineorder", "ecommerce"],
    "velocity_1h": ["velocity_1h", "velocity1h", "vel_1h", "vel1h", "velocity", "velocity_hour"],
    "velocity_24h": ["velocity_24h", "velocity24h", "vel_24h", "vel24h", "velocity_day"],
    "device_trust_score": ["device_trust_score", "device_trust", "trust_score", "devicetrust"],
    "carrier_sim_swap_age_days": ["carrier_sim_swap_age_days", "sim_swap_age_days", "sim_swap_age", "sim_age", "carriersimswapagedays"],
    "ip_country_match": ["ip_country_match", "ip_match", "country_match"],
    "two_factor_auth_success": ["two_factor_auth_success", "2fa_success", "two_factor", "twofactorauthsuccess", "2fa"],
    "is_fraud": ["is_fraud", "fraud", "class", "isfraud", "label", "target"],
}


@app.post("/api/upload-csv")
async def upload_transactions_csv(file: UploadFile = File(...)):
    """Accepts any user-provided CSV of transactions and processes each through the 3-tier cascade."""
    if STATE["engine"] is None:
        raise HTTPException(status_code=503, detail="System models not yet initialized.")

    import csv
    import io

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Automatically handle Excel BOM (utf-8-sig)
    try:
        decoded = contents.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            decoded = contents.decode("utf-8")
        except UnicodeDecodeError:
            decoded = contents.decode("latin-1")

    reader = csv.DictReader(io.StringIO(decoded))
    if not reader.fieldnames:
        raise HTTPException(status_code=400, detail="CSV has no valid header row.")

    # Build header mapping using aliases
    canonical_headers = {}
    for col in reader.fieldnames:
        if not col:
            continue
        clean_col = col.strip().lower().replace(" ", "_").replace("-", "")
        matched = False
        for canon, aliases in CSV_ALIAS_MAP.items():
            clean_aliases = [a.replace("_", "").replace("-", "") for a in aliases]
            if clean_col in clean_aliases:
                canonical_headers[col] = canon
                matched = True
                break
        if not matched:
            canonical_headers[col] = clean_col

    results = []
    engine = STATE["engine"]
    errors = []
    tier_counts = {0: 0, 1: 0, 2: 0}

    for row_idx, row in enumerate(reader):
        if row_idx >= 1500:  # Safety cap for responsive browser execution
            break

        # Remap row keys to canonical names
        clean_row = {}
        for k, v in row.items():
            if k is not None:
                canon_k = canonical_headers.get(k, k.strip().lower().replace(" ", "_"))
                clean_row[canon_k] = v

        try:
            amt = parse_smart_float(clean_row.get("amount"), default=None)
            if amt is None or amt <= 0:
                # If amount is missing or invalid, check if there's any float column
                amt = 50.0  # Safe default if not found

            dist_home = parse_smart_float(clean_row.get("distance_from_home"), default=None)
            if dist_home is None:
                # If Kaggle credit card features exist (V1..V28)
                if "v1" in clean_row:
                    dist_home = round(abs(parse_smart_float(clean_row.get("v1"), 0.0)) * 12.0 + 3.0, 1)
                else:
                    dist_home = 5.0

            dist_last = parse_smart_float(clean_row.get("distance_from_last_tx"), default=max(0.5, round(dist_home * 0.3, 1)))
            ratio_price = parse_smart_float(clean_row.get("ratio_to_median_price"), default=1.0)
            merchant = parse_smart_int(clean_row.get("merchant_category"), default=1)
            repeat = parse_smart_int(clean_row.get("repeat_retailer"), default=1)
            chip = parse_smart_int(clean_row.get("used_chip"), default=1)
            pin = parse_smart_int(clean_row.get("used_pin"), default=0)
            online = parse_smart_int(clean_row.get("online_order"), default=0)
            vel1 = parse_smart_int(clean_row.get("velocity_1h"), default=1)
            vel24 = parse_smart_int(clean_row.get("velocity_24h"), default=vel1 + 2)

            device_trust = parse_smart_opt_float(clean_row.get("device_trust_score"))
            carrier_sim = parse_smart_opt_int(clean_row.get("carrier_sim_swap_age_days"))
            ip_match = parse_smart_opt_int(clean_row.get("ip_country_match"))
            two_factor = parse_smart_opt_int(clean_row.get("two_factor_auth_success"))
            is_fraud = parse_smart_opt_int(clean_row.get("is_fraud"))

            tx = Transaction(
                transaction_id=str(clean_row.get("transaction_id") or f"CSV-TX-{1000 + row_idx}"),
                amount=max(0.01, amt),
                merchant_category=merchant,
                distance_from_home=dist_home,
                distance_from_last_tx=dist_last,
                ratio_to_median_price=ratio_price,
                repeat_retailer=repeat,
                used_chip=chip,
                used_pin=pin,
                online_order=online,
                velocity_1h=vel1,
                velocity_24h=vel24,
                device_trust_score=device_trust,
                carrier_sim_swap_age_days=carrier_sim,
                ip_country_match=ip_match,
                two_factor_auth_success=two_factor,
                is_fraud=is_fraud,
            )

            packet = engine.process_transaction(tx)
            tier_counts[packet.escalation_tier] = tier_counts.get(packet.escalation_tier, 0) + 1

            if packet.escalation_tier == 2:
                STATE["queue_manager"].enqueue(packet, tx)

            results.append(packet.model_dump())
        except Exception as e:
            errors.append(f"Row {row_idx+1}: {str(e)}")

    if not results and errors:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process CSV. Sample errors: {'; '.join(errors[:3])}. Detected headers: {list(reader.fieldnames)}"
        )

    return {
        "status": "SUCCESS",
        "processed_count": len(results),
        "tier_breakdown": tier_counts,
        "results": results,
    }


@app.get("/api/download/test-csv")
async def download_test_csv():
    """Direct download endpoint for test_transactions.csv."""
    csv_path = Path(__file__).resolve().parent.parent.parent / "test_transactions.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="test_transactions.csv not found")
    return FileResponse(
        str(csv_path),
        media_type="text/csv",
        filename="test_transactions.csv",
    )


@app.get("/api/simulate/next", response_model=DecisionPacket)
async def simulate_next_stream_transaction():
    """Streams the next transaction from the holdout set or generates an ad-hoc case."""
    if STATE["engine"] is None:
        raise HTTPException(status_code=503, detail="System models not initialized.")

    test_txs = STATE["test_txs"]
    if test_txs:
        idx = STATE["tx_stream_idx"] % len(test_txs)
        STATE["tx_stream_idx"] += 1
        tx = test_txs[idx]
    else:
        gen = TransactionDatasetGenerator(random_seed=42)
        _, _, txs = gen.generate_dataset(n_samples=10)
        tx = txs[0]

    engine = STATE["engine"]
    packet = engine.process_transaction(tx)

    if packet.escalation_tier == 2:
        STATE["queue_manager"].enqueue(packet, tx)

    return packet


@app.get("/api/queue")
async def get_pending_queue():
    """Returns pending human-in-the-loop investigation cases with full dossier briefs."""
    pending = STATE["queue_manager"].get_pending()
    briefs = []
    for item in pending:
        packet_obj = DecisionPacket(**item["packet"])
        tx_obj = Transaction(**item["transaction"])
        brief = InvestigatorExplainer.generate_case_brief(packet_obj, tx_obj)
        briefs.append({
            "queue_item": item,
            "dossier": brief,
        })
    return {
        "pending_cases": briefs,
        "stats": STATE["queue_manager"].get_queue_stats(),
    }


class ResolvePayload(BaseModel):
    transaction_id: str
    decision: str  # "APPROVE" or "DECLINE"
    analyst_id: str = "analyst_01"
    notes: str = ""


@app.post("/api/queue/resolve")
async def resolve_queue_item(payload: ResolvePayload):
    res = STATE["queue_manager"].resolve(
        payload.transaction_id,
        decision=payload.decision,
        analyst_id=payload.analyst_id,
        notes=payload.notes,
    )
    if res is None:
        raise HTTPException(status_code=404, detail="Transaction ID not found in pending queue.")

    # Seal human decision into Blockchain Audit Ledger
    if STATE["engine"] and getattr(STATE["engine"], "blockchain_ledger", None):
        STATE["engine"].blockchain_ledger.record_human_resolution(
            transaction_id=payload.transaction_id,
            decision=payload.decision,
            analyst_id=payload.analyst_id,
            notes=payload.notes,
        )

    return {"status": "RESOLVED", "case": res}


@app.get("/api/scenarios/canonical")
async def get_canonical_scenarios():
    """Returns definitions and pre-configured payloads for the 5 Canonical Research Cases."""
    return {
        "hyderabad_outlier": {
            "title": "🇮🇳 ₹85,000 Hyderabad Midnight Outlier (Proposed Research Case)",
            "description": "Customer normally spends ₹500–₹5,000 in Andhra Pradesh on Samsung S24. Suddenly: ₹85,000 to new merchant from new unknown device at 2:13 AM in Hyderabad. AI starts at 72% fraud / High uncertainty. Step 1 (Tx history) -> 88% / Med uncertainty. Step 2 (Device) -> 96% / Low uncertainty -> Auto-Blocked without human labor!",
            "canonical_mode": "two_step",
            "transaction": {
                "transaction_id": "IN-HYD-85000-MIDNIGHT",
                "amount": 85000.0,
                "merchant_category": 5,
                "distance_from_home": 165.0,
                "distance_from_last_tx": 45.0,
                "ratio_to_median_price": 34.0,
                "repeat_retailer": 0,
                "used_chip": 0,
                "used_pin": 0,
                "online_order": 1,
                "velocity_1h": 1,
                "velocity_24h": 1,
                "customer_id": "CUST-AP-4821",
                "customer_home_state": "Andhra Pradesh",
                "location_city": "Hyderabad",
                "time_of_day": "02:13 AM",
                "normal_avg_amount": 2500.0,
                "normal_device": "Samsung Galaxy S24",
                "device_fingerprint": "Unknown Android Device",
                "merchant_name": "Zenith Tech Electronics",
                "historical_merchant_tx_count": 0,
                "is_fraud": 1
            }
        },
        "case_1_confident_genuine": {
            "title": "Case 1: Confident Genuine (₹600 Routine)",
            "description": "₹600 routine local groceries, familiar merchant used 20 times, registered Samsung S24. Initial fraud risk 1%, low uncertainty. Conformal prediction set {Legit}. Resolved at Tier 0 autonomously with $0 overhead.",
            "canonical_mode": None,
            "transaction": {
                "transaction_id": "TX-CASE1-GENUINE-600",
                "amount": 600.0,
                "merchant_category": 1,
                "distance_from_home": 2.5,
                "distance_from_last_tx": 1.0,
                "ratio_to_median_price": 0.95,
                "repeat_retailer": 1,
                "used_chip": 1,
                "used_pin": 1,
                "online_order": 0,
                "velocity_1h": 1,
                "velocity_24h": 2,
                "customer_home_state": "Andhra Pradesh",
                "location_city": "Vijayawada",
                "normal_avg_amount": 2500.0,
                "normal_device": "Samsung Galaxy S24",
                "device_fingerprint": "Samsung Galaxy S24",
                "is_fraud": 0
            }
        },
        "case_2_confident_fraud": {
            "title": "Case 2: Confident Fraud (Direct High-Risk Block)",
            "description": "Obvious high-risk fraud attempt with burst velocity, foreign IP, and maximum entropy. AI outputs 98% fraud with high statistical certainty. Blocked immediately at Tier 0.",
            "canonical_mode": None,
            "transaction": {
                "transaction_id": "TX-CASE2-CONFIDENT-FRAUD",
                "amount": 145000.0,
                "merchant_category": 8,
                "distance_from_home": 620.0,
                "distance_from_last_tx": 300.0,
                "ratio_to_median_price": 58.0,
                "repeat_retailer": 0,
                "used_chip": 0,
                "used_pin": 0,
                "online_order": 1,
                "velocity_1h": 8,
                "velocity_24h": 19,
                "is_fraud": 1
            }
        },
        "case_3_step1_resolved": {
            "title": "Case 3: Uncertain Resolved by 1st Evidence (Transaction History)",
            "description": "Borderline transaction where the initial AI is uncertain. The Evidence Selector selects Transaction History (highest VoI). The spend history is completely normal (1.1x avg), collapsing uncertainty. Auto-Approved at Tier 1 with single $0.05 micro-probe.",
            "canonical_mode": "single_step",
            "transaction": {
                "transaction_id": "TX-CASE3-RESOLVED-STEP1",
                "amount": 2800.0,
                "merchant_category": 2,
                "distance_from_home": 38.0,
                "distance_from_last_tx": 12.0,
                "ratio_to_median_price": 1.1,
                "repeat_retailer": 1,
                "used_chip": 1,
                "used_pin": 0,
                "online_order": 1,
                "velocity_1h": 2,
                "velocity_24h": 3,
                "normal_avg_amount": 2500.0,
                "is_fraud": 0
            }
        },
        "case_4_step2_resolved": {
            "title": "Case 4: Uncertain Resolved by 2nd Evidence (Tx History + Device)",
            "description": "Borderline transaction where 1st evidence isn't enough to collapse uncertainty. The Evidence Selector adapts and requests 2nd evidence (Device Fingerprint), which collapses uncertainty to Low. Auto-Blocked at Tier 1.",
            "canonical_mode": "two_step",
            "transaction": {
                "transaction_id": "TX-CASE4-RESOLVED-STEP2",
                "amount": 85000.0,
                "merchant_category": 5,
                "distance_from_home": 165.0,
                "distance_from_last_tx": 45.0,
                "ratio_to_median_price": 34.0,
                "repeat_retailer": 0,
                "used_chip": 0,
                "used_pin": 0,
                "online_order": 1,
                "velocity_1h": 1,
                "velocity_24h": 1,
                "customer_home_state": "Andhra Pradesh",
                "location_city": "Hyderabad",
                "normal_avg_amount": 2500.0,
                "device_fingerprint": "Unknown Device",
                "is_fraud": 1
            }
        },
        "case_5_inconclusive_human": {
            "title": "Case 5: Inconclusive After Evidence -> Escalate to Human Investigator",
            "description": "Borderline transaction where Transaction History, Device History, and Location History all return ambiguous signals (72% -> 70% -> 68% -> 73%). Uncertainty remains high. Further evidence gathering halted to prevent waste. Escalated to Human Investigator with complete 3-step investigation trajectory.",
            "canonical_mode": "inconclusive_human",
            "transaction": {
                "transaction_id": "TX-CASE5-INCONCLUSIVE-HITL",
                "amount": 85000.0,
                "merchant_category": 5,
                "distance_from_home": 85.0,
                "distance_from_last_tx": 20.0,
                "ratio_to_median_price": 34.0,
                "repeat_retailer": 0,
                "used_chip": 0,
                "used_pin": 0,
                "online_order": 1,
                "velocity_1h": 1,
                "velocity_24h": 2,
                "normal_avg_amount": 2500.0,
                "is_fraud": 1
            }
        }
    }


@app.get("/api/audit/ledger")
async def get_blockchain_ledger(limit: int = 50):
    """Returns the immutable cryptographic investigation ledger."""
    if STATE["engine"] is None or not hasattr(STATE["engine"], "blockchain_ledger"):
        return {"blocks": [], "total": 0}
    chain = STATE["engine"].blockchain_ledger.chain
    blocks_dump = [b.model_dump() for b in reversed(chain[-limit:])]
    return {
        "blocks": blocks_dump,
        "total_blocks": len(chain),
        "head_hash": chain[-1].block_hash if chain else None,
    }


@app.get("/api/audit/verify")
async def verify_blockchain_integrity():
    """Validates the cryptographic integrity of all blocks and pointers in the chain."""
    if STATE["engine"] is None or not hasattr(STATE["engine"], "blockchain_ledger"):
        raise HTTPException(status_code=503, detail="System models not initialized.")
    return STATE["engine"].blockchain_ledger.verify_chain_integrity()


class TamperPayload(BaseModel):
    block_index: int
    fake_action: str = "TAMPERED_APPROVE"


@app.post("/api/audit/simulate-tamper")
async def simulate_blockchain_tampering(payload: TamperPayload):
    """Simulates an unauthorized modification to show cryptographic tamper detection in real time."""
    if STATE["engine"] is None or not hasattr(STATE["engine"], "blockchain_ledger"):
        raise HTTPException(status_code=503, detail="System models not initialized.")
    return STATE["engine"].blockchain_ledger.simulate_tampering(
        block_index=payload.block_index, fake_action=payload.fake_action
    )


class RestorePayload(BaseModel):
    block_index: int
    original_action: str


@app.post("/api/audit/restore")
async def restore_blockchain_block(payload: RestorePayload):
    """Restores a tampered block to verify chain recovery."""
    if STATE["engine"] is None or not hasattr(STATE["engine"], "blockchain_ledger"):
        raise HTTPException(status_code=503, detail="System models not initialized.")
    STATE["engine"].blockchain_ledger.restore_block(
        payload.block_index, payload.original_action
    )
    return {
        "status": "RESTORED",
        "verification": STATE["engine"].blockchain_ledger.verify_chain_integrity()
    }


@app.post("/api/benchmark/run")
async def run_benchmark_endpoint():
    """Triggers an empirical benchmark evaluation comparing all baselines."""
    suite = BenchmarkSuite(config=STATE["config"])
    results = suite.run_benchmark(n_samples=6000)
    STATE["benchmark_results"] = results
    return results


@app.get("/api/benchmark/latest")
async def get_latest_benchmark():
    if STATE["benchmark_results"] is None:
        # Run standard benchmark if not yet executed
        suite = BenchmarkSuite(config=STATE["config"])
        STATE["benchmark_results"] = suite.run_benchmark(n_samples=6000)
    return STATE["benchmark_results"]


class ConfigUpdatePayload(BaseModel):
    cost_false_positive: Optional[float] = None
    cost_human_review: Optional[float] = None
    cost_evidence_acquisition: Optional[float] = None
    high_value_amount_threshold: Optional[float] = None


@app.post("/api/config/update")
async def update_config_params(payload: ConfigUpdatePayload):
    costs = STATE["config"].costs
    unc = STATE["config"].uncertainty

    if payload.cost_false_positive is not None:
        costs.cost_false_positive = payload.cost_false_positive
    if payload.cost_human_review is not None:
        costs.cost_human_review = payload.cost_human_review
    if payload.cost_evidence_acquisition is not None:
        costs.cost_evidence_acquisition = payload.cost_evidence_acquisition
    if payload.high_value_amount_threshold is not None:
        unc.high_value_amount_threshold = payload.high_value_amount_threshold

    # Update engines
    if STATE["engine"]:
        STATE["engine"].cost_evaluator = CostMatrixEvaluator(costs)

    return {"status": "UPDATED", "config": STATE["config"].to_dict()}
