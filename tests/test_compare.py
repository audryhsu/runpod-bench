#!/usr/bin/env python3
"""Unit tests for compare.py using synthetic fixture data."""

import io
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))
from compare import pct_delta, color_delta, RESET

FIXTURES = Path(__file__).parent / "fixtures"
EC2 = FIXTURES / "ec2-direct" / "2026-05-06T12:00:00Z"
POD = FIXTURES / "runpod-pod" / "2026-05-06T14:00:00Z"


class TestPctDelta(unittest.TestCase):
    def test_positive_delta(self):
        # treatment 10% higher than baseline
        result = pct_delta(100.0, 110.0)
        self.assertAlmostEqual(result, 10.0)

    def test_negative_delta(self):
        result = pct_delta(100.0, 90.0)
        self.assertAlmostEqual(result, -10.0)

    def test_zero_delta(self):
        result = pct_delta(100.0, 100.0)
        self.assertAlmostEqual(result, 0.0)

    def test_none_baseline_returns_none(self):
        self.assertIsNone(pct_delta(None, 100.0))

    def test_none_treatment_returns_none(self):
        self.assertIsNone(pct_delta(100.0, None))

    def test_zero_baseline_returns_none(self):
        self.assertIsNone(pct_delta(0, 100.0))

    def test_small_delta(self):
        # 1% improvement
        result = pct_delta(200.0, 202.0)
        self.assertAlmostEqual(result, 1.0)


class TestColorDelta(unittest.TestCase):
    def test_none_returns_na(self):
        self.assertEqual(color_delta(None), "n/a")

    def test_small_delta_is_green(self):
        # < 5% -- green regardless of direction
        result = color_delta(2.0, higher_is_worse=True)
        self.assertIn("\033[92m", result)  # GREEN

    def test_medium_delta_is_yellow(self):
        # 5-15%
        result = color_delta(10.0, higher_is_worse=True)
        self.assertIn("\033[93m", result)  # YELLOW

    def test_large_bad_delta_is_red(self):
        # >15% latency increase is bad
        result = color_delta(20.0, higher_is_worse=True)
        self.assertIn("\033[91m", result)  # RED

    def test_large_good_delta_is_green(self):
        # >15% throughput increase is good (higher_is_worse=False, delta > 0)
        result = color_delta(20.0, higher_is_worse=False)
        self.assertIn("\033[92m", result)  # GREEN

    def test_output_ends_with_reset(self):
        result = color_delta(5.0)
        self.assertIn(RESET, result)

    def test_positive_delta_has_plus_sign(self):
        result = color_delta(3.0)
        self.assertIn("+3.0%", result)

    def test_negative_delta_no_plus_sign(self):
        result = color_delta(-3.0)
        self.assertNotIn("+", result.replace(RESET, "").replace("\033[92m", ""))


class TestEndToEnd(unittest.TestCase):
    """Run compare.py as subprocess against fixture summaries."""

    @classmethod
    def setUpClass(cls):
        """Generate summaries for both fixture dirs before running compare tests."""
        for results_dir in [EC2, POD]:
            subprocess.run(
                ["python3", "summarize.py", str(results_dir)],
                cwd=Path(__file__).parent.parent,
                capture_output=True,
            )

    def _run_compare(self, base_dir, treat_dir):
        return subprocess.run(
            ["python3", "compare.py", str(base_dir), str(treat_dir)],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
        )

    def test_exits_zero(self):
        result = self._run_compare(EC2, POD)
        self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_output_contains_manifest_section(self):
        result = self._run_compare(EC2, POD)
        self.assertIn("Manifest Comparison", result.stdout)

    def test_output_contains_vllm_section(self):
        result = self._run_compare(EC2, POD)
        self.assertIn("vLLM Serving", result.stdout)

    def test_output_contains_cold_start_section(self):
        result = self._run_compare(EC2, POD)
        self.assertIn("Cold Start", result.stdout)

    def test_output_contains_fio_section(self):
        result = self._run_compare(EC2, POD)
        self.assertIn("Storage", result.stdout)

    def test_output_contains_cpu_section(self):
        result = self._run_compare(EC2, POD)
        self.assertIn("CPU", result.stdout)

    def test_env_names_in_output(self):
        result = self._run_compare(EC2, POD)
        self.assertIn("ec2-direct", result.stdout)
        self.assertIn("runpod-pod", result.stdout)

    def test_cgroup_throttle_delta_visible(self):
        # Pod fixture has nr_throttled_delta=47, ec2 has 0 -- delta should appear
        result = self._run_compare(EC2, POD)
        self.assertIn("throttled", result.stdout)

    def test_missing_args_exits_nonzero(self):
        result = subprocess.run(
            ["python3", "compare.py"],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)

    def test_missing_summary_json_still_runs(self):
        """compare.py should not crash even if summary.json is missing -- it warns and continues."""
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            # Create a dir with only a manifest, no summary.json
            manifest = {"hostname": "test", "gpu_model": "NVIDIA L4", "gpu_count": 8,
                        "vllm_version": "0.8.3"}
            manifest_path = Path(tmp) / "manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(manifest, f)
            result = self._run_compare(tmp, str(EC2))
            # Should not crash -- partial output is acceptable
            self.assertEqual(result.returncode, 0)

    def test_vllm_version_mismatch_warning(self):
        """If vLLM versions differ between runs, compare.py should warn."""
        import tempfile, shutil
        with tempfile.TemporaryDirectory() as tmp:
            # Copy ec2 fixture and bump vllm_version in manifest
            shutil.copytree(str(EC2), tmp + "/run", dirs_exist_ok=True)
            manifest_path = Path(tmp) / "run" / "manifest.json"
            with open(manifest_path) as f:
                m = json.load(f)
            m["vllm_version"] = "0.9.0"  # different version
            with open(manifest_path, "w") as f:
                json.dump(m, f)
            # Generate summary for modified run
            subprocess.run(
                ["python3", "summarize.py", tmp + "/run"],
                cwd=Path(__file__).parent.parent,
                capture_output=True,
            )
            result = self._run_compare(EC2, tmp + "/run")
            self.assertIn("WARNING", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
