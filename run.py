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


def run_benchmark_cli(samples: int = 10000):
    from safe_escalate.eval.benchmark import BenchmarkSuite

    print("=" * 80)
    print("SafeEscalate Empirical Research Benchmark")
    print(f"Generating and evaluating dataset ({samples} total samples)...")
    print("=" * 80)

    suite = BenchmarkSuite()
    results = suite.run_benchmark(n_samples=samples)

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


def run_server_cli(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    import uvicorn

    print(f"Starting SafeEscalate Server at http://{host}:{port}")
    uvicorn.run("safe_escalate.web.app:app", host=host, port=port, reload=reload)


def main():
    parser = argparse.ArgumentParser(description="SafeEscalate CLI Runner")
    parser.add_argument(
        "action",
        nargs="?",
        default="serve",
        choices=["serve", "benchmark", "test"],
        help="Action to perform: 'serve' to launch UI/API, 'benchmark' to run evaluations.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host address for web server")
    parser.add_argument("--port", type=int, default=8000, help="Port for web server")
    parser.add_argument("--samples", type=int, default=8000, help="Sample size for benchmark")

    args = parser.parse_args()

    if args.action == "benchmark":
        run_benchmark_cli(samples=args.samples)
    elif args.action == "test":
        import unittest
        loader = unittest.TestLoader()
        start_dir = os.path.join(CURRENT_DIR, "tests")
        suite = loader.discover(start_dir)
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)
    else:
        run_server_cli(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
