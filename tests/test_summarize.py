#!/usr/bin/env python3
"""Unit tests for summarize.py using synthetic fixture data."""

import json
import sys
import unittest
from pathlib import Path

# Add parent dir so we can import from summarize.py directly
sys.path.insert(0, str(Path(__file__).parent.parent))
from summarize import (
    percentile,
    summarize_vllm_serving,
    summarize_cold_start,
    summarize_fio,
    summarize_cpu,
    format_summary_txt,
)

FIXTURES = Path(__file__).parent / "fixtures"
EC2 = FIXTURES / "ec2-direct" / "2026-05-06T12:00:00Z"
POD = FIXTURES / "runpod-pod" / "2026-05-06T14:00:00Z"
WARMUP = 1  # matches fixture config


class TestPercentile(unittest.TestCase):
    def test_median_of_sorted(self):
        self.assertAlmostEqual(percentile([1, 2, 3, 4, 5], 50), 3.0)

    def test_p95_of_list(self):
        data = list(range(1, 101))  # 1..100
        self.assertAlmostEqual(percentile(data, 95), 95.05, places=1)

    def test_single_element(self):
        self.assertEqual(percentile([42], 95), 42)

    def test_empty(self):
        self.assertIsNone(percentile([], 50))


class TestSummarizeVllmServing(unittest.TestCase):
    def setUp(self):
        self.summary = summarize_vllm_serving(EC2, WARMUP)

    def test_returns_dict_with_rates(self):
        self.assertIsNotNone(self.summary)
        self.assertIn("1", self.summary)
        self.assertIn("4", self.summary)
        self.assertIn("inf", self.summary)

    def test_warmup_discarded(self):
        # 5 measured runs + 1 warmup = 6 files; should have 5 values
        values = self.summary["1"]["request_throughput"]["values"]
        self.assertEqual(len(values), 5, "Warmup run should be discarded")

    def test_request_throughput_rate1(self):
        rt = self.summary["1"]["request_throughput"]
        self.assertAlmostEqual(rt["median"], 1.0, places=1)

    def test_median_ttft_rate1(self):
        ttft = self.summary["1"]["median_ttft_ms"]
        # All runs are ~42ms, median should be ~42
        self.assertGreater(ttft["median"], 41.0)
        self.assertLess(ttft["median"], 43.0)

    def test_output_throughput_present(self):
        self.assertIn("output_throughput", self.summary["1"])

    def test_rate_inf_present(self):
        inf_data = self.summary["inf"]
        self.assertIn("request_throughput", inf_data)
        self.assertAlmostEqual(inf_data["request_throughput"]["median"], 18.5, places=0)

    def test_missing_dir_returns_none(self):
        result = summarize_vllm_serving(Path("/nonexistent"), WARMUP)
        self.assertIsNone(result)


class TestSummarizeColdStart(unittest.TestCase):
    def setUp(self):
        self.summary = summarize_cold_start(EC2, WARMUP)

    def test_returns_dict(self):
        self.assertIsNotNone(self.summary)
        self.assertIn("median_s", self.summary)
        self.assertIn("p95_s", self.summary)
        self.assertIn("values", self.summary)

    def test_warmup_discarded(self):
        self.assertEqual(len(self.summary["values"]), 5)

    def test_median_in_expected_range(self):
        # fixture durations: 28.1, 28.3, 27.9, 28.5, 28.2 → median ~28.2
        self.assertGreater(self.summary["median_s"], 27.5)
        self.assertLess(self.summary["median_s"], 29.0)

    def test_p95_gte_median(self):
        self.assertGreaterEqual(self.summary["p95_s"], self.summary["median_s"])

    def test_missing_dir_returns_none(self):
        result = summarize_cold_start(Path("/nonexistent"), WARMUP)
        self.assertIsNone(result)


class TestSummarizeFio(unittest.TestCase):
    def setUp(self):
        self.summary = summarize_fio(EC2)

    def test_all_profiles_present(self):
        self.assertIsNotNone(self.summary)
        for profile in ["seq_read", "seq_write", "rand_read", "rand_write"]:
            self.assertIn(profile, self.summary, f"Missing profile: {profile}")

    def test_seq_read_bandwidth(self):
        # 3355443200 bytes / (1024*1024) = ~3200 MB/s
        bw = self.summary["seq_read"]["bw_mbs"]
        self.assertAlmostEqual(bw, 3200.0, delta=10)

    def test_seq_write_bandwidth(self):
        bw = self.summary["seq_write"]["bw_mbs"]
        self.assertAlmostEqual(bw, 2000.0, delta=10)

    def test_latency_p99_gt_p50(self):
        for profile in self.summary:
            p50 = self.summary[profile]["lat_p50_ms"]
            p99 = self.summary[profile]["lat_p99_ms"]
            self.assertGreaterEqual(p99, p50, f"{profile}: p99 should be >= p50")

    def test_iops_positive(self):
        for profile in self.summary:
            self.assertGreater(self.summary[profile]["iops"], 0)

    def test_missing_dir_returns_none(self):
        result = summarize_fio(Path("/nonexistent"))
        self.assertIsNone(result)


class TestSummarizeCpu(unittest.TestCase):
    def setUp(self):
        self.summary = summarize_cpu(EC2)

    def test_sysbench_present(self):
        self.assertIsNotNone(self.summary)
        self.assertIn("sysbench", self.summary)

    def test_sysbench_fields(self):
        sb = self.summary["sysbench"]
        self.assertIn("events_per_sec", sb)
        self.assertIn("latency_avg_ms", sb)
        self.assertIn("latency_p95_ms", sb)
        self.assertAlmostEqual(sb["events_per_sec"], 12400.5, delta=1)

    def test_cgroup_throttle_present(self):
        self.assertIn("cgroup_throttle", self.summary)

    def test_cgroup_zero_for_baseline(self):
        cg = self.summary["cgroup_throttle"]
        self.assertEqual(cg["nr_throttled_delta"], 0)
        self.assertEqual(cg["throttled_usec_delta"], 0)

    def test_pod_cgroup_shows_throttling(self):
        pod_cpu = summarize_cpu(POD)
        self.assertIsNotNone(pod_cpu)
        cg = pod_cpu["cgroup_throttle"]
        self.assertGreater(cg["nr_throttled_delta"], 0)
        self.assertGreater(cg["throttled_usec_delta"], 0)

    def test_missing_dir_returns_none(self):
        result = summarize_cpu(Path("/nonexistent"))
        self.assertIsNone(result)


class TestFormatSummaryTxt(unittest.TestCase):
    def setUp(self):
        # Build a minimal summary dict
        self.summary = {
            "vllm_serving": {
                "1": {
                    "request_throughput": {"median": 1.0, "p95": 1.0, "values": [1.0]},
                    "output_throughput": {"median": 128.0, "p95": 128.0, "values": [128.0]},
                    "median_ttft_ms": {"median": 42.0, "p95": 45.0, "values": [42.0]},
                    "p99_ttft_ms": {"median": 57.0, "p95": 58.0, "values": [57.0]},
                    "median_tpot_ms": {"median": 8.2, "p95": 8.3, "values": [8.2]},
                }
            },
            "cold_start": {"median_s": 28.2, "p95_s": 28.5, "values": [28.2]},
            "fio": {
                "seq_read": {"bw_mbs": 3200.0, "iops": 3200.0, "lat_p50_ms": 0.3, "lat_p99_ms": 1.2}
            },
            "cpu": {
                "sysbench": {"events_per_sec": 12400.5, "latency_avg_ms": 0.08, "latency_p95_ms": 0.10},
                "cgroup_throttle": {"nr_throttled_delta": 0, "throttled_usec_delta": 0},
            },
        }

    def test_output_is_string(self):
        txt = format_summary_txt(self.summary)
        self.assertIsInstance(txt, str)

    def test_contains_section_headers(self):
        txt = format_summary_txt(self.summary)
        self.assertIn("vLLM Serving", txt)
        self.assertIn("Cold Start", txt)
        self.assertIn("Storage", txt)
        self.assertIn("CPU", txt)

    def test_cold_start_values_in_output(self):
        txt = format_summary_txt(self.summary)
        self.assertIn("28.2", txt)

    def test_empty_summary_produces_string(self):
        txt = format_summary_txt({})
        self.assertIsInstance(txt, str)

    def test_vllm_rate_in_output(self):
        txt = format_summary_txt(self.summary)
        self.assertIn("1", txt)  # rate 1 should appear


class TestEndToEnd(unittest.TestCase):
    """Run summarize.py as a subprocess against the fixture dirs and check output."""

    def _run_summarize(self, results_dir):
        import subprocess
        result = subprocess.run(
            ["python3", "summarize.py", str(results_dir)],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
        )
        return result

    def test_ec2_direct_summarize_exits_zero(self):
        result = self._run_summarize(EC2)
        self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_pod_summarize_exits_zero(self):
        result = self._run_summarize(POD)
        self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_ec2_summary_json_written(self):
        self._run_summarize(EC2)
        summary_path = EC2 / "summary.json"
        self.assertTrue(summary_path.exists())
        with open(summary_path) as f:
            data = json.load(f)
        self.assertIn("vllm_serving", data)
        self.assertIn("cold_start", data)
        self.assertIn("fio", data)
        self.assertIn("cpu", data)

    def test_ec2_summary_txt_written(self):
        self._run_summarize(EC2)
        txt_path = EC2 / "summary.txt"
        self.assertTrue(txt_path.exists())
        content = txt_path.read_text()
        self.assertIn("vLLM Serving", content)

    def test_bad_dir_exits_nonzero(self):
        result = self._run_summarize(Path("/nonexistent/path"))
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
