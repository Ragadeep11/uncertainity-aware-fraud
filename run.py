"""
Unified SafeEscalate CLI runner: Start web application or execute empirical benchmarks.
"""

import argparse
import sys
import os

# Ensure safe_escalate is importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)


def run_benchmark_cli(samples: int = 50000, dataset: str = "kaggle"):
    from safe_escalate.eval.benchmark import BenchmarkSuite

    print("=" * 80)
    print("SafeEscalate Empirical Research Benchmark")
    print(f"Dataset: {dataset.upper()} | Samples: {samples:,}")
    print("=" * 80)

    suite = BenchmarkSuite(dataset_type=dataset)
    results = suite.run_benchmark(n_samples=samples, dataset_type=dataset)

    print("\n--- Conformal Calibration Metrics ---")
    cal = results["conformal_calibration"]
    print(f"Target Coverage:    {cal['target_coverage']*100:.1f}%")
    print(f"Empirical Coverage: {cal['empirical_coverage']*100:.1f}%")
    print(f"Ambiguity Rate:     {cal['ambiguity_rate']*100:.1f}%")
    print(f"Quantile (q_hat):   {cal['q_hat']:.4f}")

    print("\n--- Comparative Evaluation Across Decision Policies ---")
    print(
        f"{'Model / Policy':<35} | {'Total Cost':<12} | {'Fraud Loss':<12} | {'Human Labor':<12} | {'Reviews':<8} | {'Recall':<8}"
    )
    print("-" * 105)

    for key, m in results["models"].items():
        print(
            f"{m['name']:<35} | ${m['total_cost']:<11,.2f} | ${m['fraud_loss']:<11,.2f} | ${m['human_cost']:<11,.2f} | {m['human_reviews']:<8} | {m['fraud_recall_pct']:<7.1f}%"
        )
    print("=" * 105)

    se = results["models"]["SafeEscalate_Proposed"]
    print(f"\n[KEY FINDING] Human Workload Reduction vs Direct Abstention: {se['human_workload_reduction_pct']}%")
    print(f"[KEY FINDING] Total Cost Savings vs Static Single Threshold: {se['cost_savings_pct']}%\n")


def run_interactive_cli():
    """Allows user to enter a transaction interactively from the command line."""
    from safe_escalate.eval.benchmark import BenchmarkSuite
    from safe_escalate.data.schema import Transaction

    print("\n" + "=" * 65)
    print("SafeEscalate Interactive Transaction Evaluator")
    print("Initializing engine and calibrating conformal prediction sets...")
    print("=" * 65)

    suite = BenchmarkSuite()
    engine, _, cal = suite.prepare_experiment(n_samples=4000)

    print(f"Engine Ready! Conformal Target Coverage: 95.0% (q_hat = {cal['q_hat']:.4f})\n")

    while True:
        print("-" * 65)
        print("Enter Transaction Details (press Enter to use default in brackets):")
        try:
            amt_str = input("  Amount in USD [$120.00]: ").strip()
            amount = float(amt_str) if amt_str else 120.00

            dist_str = input("  Distance from home in miles [5.0]: ").strip()
            dist_home = float(dist_str) if dist_str else 5.0

            ratio_str = input("  Ratio to cardholder median price [1.1]: ").strip()
            ratio_median = float(ratio_str) if ratio_str else 1.1

            online_str = input("  Is this an Online Order? (y/n) [n]: ").strip().lower()
            online = 1 if online_str.startswith("y") else 0

            chip_str = input("  Was physical EMV Chip used? (y/n) [y]: ").strip().lower()
            chip = 0 if chip_str.startswith("n") else 1

            repeat_str = input("  Repeat/Frequent retailer? (y/n) [y]: ").strip().lower()
            repeat = 0 if repeat_str.startswith("n") else 1

            vel_str = input("  Velocity in last 1 hour (tx count) [1]: ").strip()
            vel_1h = int(vel_str) if vel_str else 1

        except (ValueError, KeyboardInterrupt):
            print("\nExiting interactive mode.")
            break

        tx = Transaction(
            transaction_id=f"CLI-TX-{int(time.time())}",
            amount=amount,
            merchant_category=1,
            distance_from_home=dist_home,
            distance_from_last_tx=max(0.5, dist_home * 0.4),
            ratio_to_median_price=ratio_median,
            repeat_retailer=repeat,
            used_chip=chip,
            used_pin=0,
            online_order=online,
            velocity_1h=vel_1h,
            velocity_24h=vel_1h + 2,
        )

        packet = engine.process_transaction(tx)

        print("\n>>> CASCASE EVALUATION AUDIT DOSSIER <<<")
        print(f"  Transaction ID:       {packet.transaction_id}")
        print(f"  Amount:               ${packet.amount:.2f}")
        print(f"  Base Model P(Fraud):  {packet.p_fraud_initial*100:.1f}%")
        print(f"  Conformal Set (95%):  {packet.conformal_set}")
        print(f"  Aleatoric / Epistemic: {packet.aleatoric_uncertainty:.2f} / {packet.epistemic_uncertainty:.2f}")
        print(f"  Escalation Tier:      Tier {packet.escalation_tier}")
        print(f"  FINAL ACTION:         {packet.final_action}")
        if packet.evidence_collected:
            ev = packet.evidence_collected
            s2fa = "Passed" if ev["two_factor_auth_success"] == 1 else "Failed / Timeout"
            print(f"  Dynamic 2FA Result:   {s2fa} (Device Trust: {ev['device_trust_score']:.2f})")
        print(f"  System Rationale:     {packet.investigator_rationale}")
        print("-" * 65 + "\n")

        cont = input("Evaluate another transaction? (y/n) [y]: ").strip().lower()
        if cont.startswith("n"):
            break


def run_server_cli(host: str = "127.0.0.1", port: int = 8000, reload: bool = False, dataset: str = "kaggle"):
    import uvicorn

    os.environ["SAFE_ESCALATE_DATASET"] = dataset
    print(f"Starting SafeEscalate Server at http://{host}:{port} (Dataset: {dataset.upper()})")
    uvicorn.run("safe_escalate.web.app:app", host=host, port=port, reload=reload)


def main():
    import time
    parser = argparse.ArgumentParser(description="SafeEscalate CLI Runner")
    parser.add_argument(
        "action",
        nargs="?",
        default="serve",
        choices=["serve", "benchmark", "test", "interactive"],
        help="Action to perform: 'serve' to launch UI/API, 'benchmark' to run evaluations, 'interactive' for terminal inputs.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host address for web server")
    parser.add_argument("--port", type=int, default=8000, help="Port for web server")
    parser.add_argument("--samples", type=int, default=50000, help="Sample size for benchmark or training")
    parser.add_argument(
        "--dataset",
        choices=["kaggle", "synthetic"],
        default="kaggle",
        help="Dataset to train models on: 'kaggle' (real European cardholders, default) or 'synthetic'.",
    )

    args = parser.parse_args()

    if args.action == "benchmark":
        run_benchmark_cli(samples=args.samples, dataset=args.dataset)
    elif args.action == "test":
        import unittest
        loader = unittest.TestLoader()
        start_dir = os.path.join(CURRENT_DIR, "tests")
        suite = loader.discover(start_dir)
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)
    elif args.action == "interactive":
        run_interactive_cli()
    else:
        run_server_cli(host=args.host, port=args.port, dataset=args.dataset)


if __name__ == "__main__":
    main()
