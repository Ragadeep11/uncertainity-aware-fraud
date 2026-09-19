"""
Unit tests for the Research Experiment Suite (Experiments A-E, Ablation, Statistical Significance, Blockchain Benchmarks).
"""

import unittest
from safe_escalate.config import AppConfig
from safe_escalate.eval.research_experiments import ResearchExperimentSuite


class TestResearchExperimentSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = AppConfig()
        cls.suite = ResearchExperimentSuite(config=cls.config, dataset_type="synthetic")
        cls.engine, cls.X_train, cls.y_train, cls.X_cal, cls.y_cal, cls.X_test, cls.y_test, cls.test_txs = (
            cls.suite.prepare_data(n_samples=300)
        )

    def test_experiment_a_baselines(self):
        """Validates standard ML baselines on fraud metrics without arbitrary targets."""
        res_a = self.suite.run_experiment_a(
            self.X_train, self.y_train, self.X_test, self.y_test
        )
        self.assertIn("Logistic_Regression", res_a)
        self.assertIn("Random_Forest", res_a)
        self.assertIn("Hist_Gradient_Boosting", res_a)

        for name, metrics in res_a.items():
            self.assertIn("precision", metrics)
            self.assertIn("recall", metrics)
            self.assertIn("f1_score", metrics)
            self.assertIn("pr_auc", metrics)
            self.assertIn("roc_auc", metrics)
            self.assertGreaterEqual(metrics["pr_auc"], 0.0)
            self.assertLessEqual(metrics["pr_auc"], 1.0)

    def test_experiment_b_calibration_and_uncertainty(self):
        """Validates ECE, Brier score, and conformal prediction coverage."""
        res_b = self.suite.run_experiment_b(
            self.engine.base_classifier,
            self.engine.conformal_predictor,
            self.engine.uncertainty_estimator,
            self.X_test,
            self.y_test,
        )
        self.assertIn("expected_calibration_error_ece", res_b)
        self.assertIn("brier_score", res_b)
        self.assertIn("empirical_conformal_coverage", res_b)
        self.assertIn("conformal_set_distribution", res_b)
        self.assertIn("error_rate_by_uncertainty_tier", res_b)
        self.assertGreaterEqual(res_b["empirical_conformal_coverage"], 0.70)

    def test_experiment_c_direct_hitl(self):
        """Validates existing dual-threshold direct escalation."""
        res_c = self.suite.run_experiment_c(self.engine, self.test_txs)
        self.assertIn("human_review_rate_pct", res_c)
        self.assertIn("fraud_recall_pct", res_c)
        self.assertEqual(res_c["avg_evidence_checks"], 0.0)

    def test_experiment_d_fixed_evidence(self):
        """Validates fixed-order investigation sequence."""
        res_d = self.suite.run_experiment_d(self.engine, self.test_txs)
        self.assertIn("avg_evidence_checks", res_d)
        self.assertGreaterEqual(res_d["avg_evidence_checks"], 0.0)

    def test_experiment_e_proposed_method(self):
        """Validates our proposed adaptive VoI method."""
        res_e = self.suite.run_experiment_e(self.engine, self.test_txs)
        self.assertIn("human_review_rate_pct", res_e)
        self.assertIn("avg_evidence_checks", res_e)
        self.assertIn("cost_per_tx", res_e)

    def test_ablation_study(self):
        """Validates component ablation impacts."""
        ablation = self.suite.run_ablation_study(self.engine, self.test_txs)
        self.assertIn("Full_Proposed_Model", ablation)
        self.assertIn("Ablation_1_No_Uncertainty", ablation)
        self.assertIn("Ablation_2_No_Cost", ablation)
        self.assertIn("Ablation_3_No_Adaptive", ablation)
        self.assertIn("Ablation_4_No_Stopping", ablation)

    def test_statistical_significance_tests(self):
        """Validates bootstrap confidence intervals and paired hypothesis testing."""
        stats_res = self.suite.run_statistical_significance_tests(
            self.engine, self.test_txs, n_bootstraps=5
        )
        self.assertIn("confidence_intervals_95", stats_res)
        self.assertIn("hypothesis_testing", stats_res)
        self.assertIn("proposed_vs_dual_threshold_human_workload", stats_res["hypothesis_testing"])

    def test_blockchain_benchmark(self):
        """Validates blockchain performance metrics against relational logs."""
        bench = self.engine.blockchain_ledger.benchmark_performance(n_blocks=10)
        self.assertIn("write_latency_ms", bench)
        self.assertIn("verification_latency_ms", bench)
        self.assertIn("throughput_blocks_per_sec", bench)
        self.assertGreater(bench["throughput_blocks_per_sec"], 100)


if __name__ == "__main__":
    unittest.main()
