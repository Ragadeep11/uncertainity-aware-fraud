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


def initialize_system(n_samples: int = 10000):
    """Pre-trains base models, conformal predictor, and validators."""
    print("Training SafeEscalate core models and conformal calibration set...")
    suite = BenchmarkSuite(config=STATE["config"])
    engine, test_txs, cal_metrics = suite.prepare_experiment(n_samples=n_samples)
    STATE["engine"] = engine
    STATE["test_txs"] = test_txs
    STATE["conformal_metrics"] = cal_metrics
    print(f"System initialized with {len(test_txs)} test transactions ready for evaluation.")


@app.on_event("startup")
async def startup_event():
    initialize_system(n_samples=10000)


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
        "conformal_q_hat": engine.conformal_predictor.q_hat,
        "conformal_alpha": engine.conformal_predictor.alpha,
        "conformal_target_coverage": 1.0 - engine.conformal_predictor.alpha,
        "queue_stats": STATE["queue_manager"].get_queue_stats(),
        "config": STATE["config"].to_dict(),
    }


@app.post("/api/predict", response_model=DecisionPacket)
async def process_transaction(tx: Transaction):
    if STATE["engine"] is None:
        raise HTTPException(status_code=503, detail="System models not yet initialized.")

    engine = STATE["engine"]
    packet = engine.process_transaction(tx)

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
    return {"status": "RESOLVED", "case": res}


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
