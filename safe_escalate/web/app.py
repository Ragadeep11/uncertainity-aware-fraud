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


@app.post("/api/upload-csv")
async def upload_transactions_csv(file: UploadFile = File(...)):
    """Accepts a user-provided CSV file of transactions and processes each through the cascade."""
    if STATE["engine"] is None:
        raise HTTPException(status_code=503, detail="System models not yet initialized.")

    import csv
    import io

    contents = await file.read()
    try:
        decoded = contents.decode("utf-8")
    except UnicodeDecodeError:
        decoded = contents.decode("latin-1")

    reader = csv.DictReader(io.StringIO(decoded))
    results = []
    engine = STATE["engine"]

    for row_idx, row in enumerate(reader):
        try:
            tx = Transaction(
                transaction_id=row.get("transaction_id", f"USER-TX-{1000 + row_idx}"),
                amount=float(row.get("amount", 50.0)),
                merchant_category=int(row.get("merchant_category", 1)),
                distance_from_home=float(row.get("distance_from_home", 5.0)),
                distance_from_last_tx=float(row.get("distance_from_last_tx", 1.0)),
                ratio_to_median_price=float(row.get("ratio_to_median_price", 1.0)),
                repeat_retailer=int(row.get("repeat_retailer", 1)),
                used_chip=int(row.get("used_chip", 1)),
                used_pin=int(row.get("used_pin", 0)),
                online_order=int(row.get("online_order", 0)),
                velocity_1h=int(row.get("velocity_1h", 0)),
                velocity_24h=int(row.get("velocity_24h", 1)),
                device_trust_score=float(row["device_trust_score"]) if "device_trust_score" in row and row["device_trust_score"] != "" else None,
                carrier_sim_swap_age_days=int(row["carrier_sim_swap_age_days"]) if "carrier_sim_swap_age_days" in row and row["carrier_sim_swap_age_days"] != "" else None,
                ip_country_match=int(row["ip_country_match"]) if "ip_country_match" in row and row["ip_country_match"] != "" else None,
                two_factor_auth_success=int(row["two_factor_auth_success"]) if "two_factor_auth_success" in row and row["two_factor_auth_success"] != "" else None,
                is_fraud=int(row["is_fraud"]) if "is_fraud" in row and row["is_fraud"] != "" else None,
            )
            packet = engine.process_transaction(tx)
            if packet.escalation_tier == 2:
                STATE["queue_manager"].enqueue(packet, tx)
            results.append(packet.model_dump())
        except Exception as e:
            continue

    return {
        "processed_count": len(results),
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
