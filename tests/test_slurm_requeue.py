"""Slurm walltime requeue helpers (pointcept/utils/slurm_requeue.py, scripts/jz/*)."""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pointcept.utils import slurm_requeue
from pointcept.utils.runtime_state import (
    flush_runtime_segment_from_env,
    init_runtime_segment,
)

REPO = Path(__file__).resolve().parents[1]


class TestParseDuration(unittest.TestCase):
    def test_formats(self):
        parse = slurm_requeue.parse_slurm_duration_to_seconds
        self.assertEqual(parse("1-02:03:04"), ((24 + 2) * 60 + 3) * 60 + 4)
        self.assertEqual(parse("19:50:00"), 19 * 3600 + 50 * 60)
        self.assertEqual(parse("59:30"), 59 * 60 + 30)
        self.assertIsNone(parse("UNLIMITED"))
        self.assertIsNone(parse(""))


class TestInstall(unittest.TestCase):
    def test_noop_without_opt_in(self):
        env = {k: v for k, v in os.environ.items() if k != "POINTCEPT_SLURM_REQUEUE"}
        env["SLURM_JOB_ID"] = "1"
        with mock.patch.dict(os.environ, env, clear=True), mock.patch(
            "signal.signal"
        ) as sig:
            slurm_requeue.install_slurm_timeout_requeue_handler()
            sig.assert_not_called()

    def test_runtime_flushed_from_job_dir(self):
        with tempfile.TemporaryDirectory() as d:
            init_runtime_segment(d)
            with mock.patch.dict(os.environ, {"JOB_DIR": d}):
                self.assertIsNotNone(flush_runtime_segment_from_env())


class TestCrashRetry(unittest.TestCase):
    def _retry(self, job_dir, duration, env=None):
        cmd = f'. scripts/jz/slurm_crash_retry.sh; should_retry_after_crash {duration} "{job_dir}"'
        return subprocess.run(
            ["sh", "-c", cmd], cwd=REPO, capture_output=True, env={**os.environ, **(env or {})}
        ).returncode

    def test_quick_crashes_bounded(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual([self._retry(d, 5) for _ in range(4)], [0, 0, 0, 1])
            # a long attempt resets the budget
            self.assertEqual(self._retry(d, 3600), 0)

    def test_no_retry_after_requeue(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, ".requeue_triggered").touch()
            self.assertEqual(self._retry(d, 5), 1)


class TestLaunchers(unittest.TestCase):
    def test_shell_syntax(self):
        for name in ["job.slurm", "sonata_pretrain.slurm", "submit.sh", "sonata_probe.sh"]:
            subprocess.run(["bash", "-n", f"scripts/jz/{name}"], cwd=REPO, check=True)
        for name in ["slurm_requeue_trap.sh", "slurm_requeue_watchdog.sh", "slurm_crash_retry.sh"]:
            subprocess.run(["sh", "-n", f"scripts/jz/{name}"], cwd=REPO, check=True)

    def test_slurm_scripts_requeue(self):
        for name in ["job.slurm", "sonata_pretrain.slurm"]:
            text = (REPO / "scripts/jz" / name).read_text()
            for needle in [
                "#SBATCH --signal=B:USR1@120",
                "#SBATCH --requeue",
                "#SBATCH --open-mode=append",
                "export POINTCEPT_SLURM_REQUEUE=1",
                ". scripts/jz/slurm_requeue_trap.sh",
                "sh scripts/jz/slurm_requeue_watchdog.sh &",
                "should_retry_after_crash",
                'rm -f "$RUN_DIR/.requeue_triggered"',
            ]:
                self.assertIn(needle, text, f"{name}: {needle}")


if __name__ == "__main__":
    unittest.main()
