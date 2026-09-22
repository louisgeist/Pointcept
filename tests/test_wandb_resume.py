"""Tests for W&B resume ``_step`` bumping (no live W&B required).

Run with: PYTHONPATH=./ pytest tests/test_wandb_resume.py
"""

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from pointcept.utils.wandb_resume import (
    bump_wandb_run_step,
    bump_wandb_step_on_resume,
    compute_resume_step,
    fetch_last_history_step,
    read_local_last_step,
    write_local_last_step,
)


class TestComputeResumeStep(unittest.TestCase):
    def test_unknown_history_keeps_local_step(self):
        self.assertEqual(compute_resume_step(21, None), 21)

    def test_bumps_past_server_history(self):
        self.assertEqual(compute_resume_step(21, 752), 753)

    def test_keeps_local_step_when_already_ahead(self):
        self.assertEqual(compute_resume_step(800, 752), 800)

    def test_equal_to_last_history_advances_by_one(self):
        self.assertEqual(compute_resume_step(752, 752), 753)

    def test_empty_history(self):
        self.assertEqual(compute_resume_step(0, -1), 0)

    def test_none_local_step_treated_as_zero(self):
        self.assertEqual(compute_resume_step(None, 10), 11)


class _FakeWandbRun:
    def __init__(self, step):
        self._step = step

    @property
    def step(self):
        return self._step


class _FakeLogger:
    def __init__(self):
        self.infos = []
        self.warnings = []

    def info(self, msg, *args):
        self.infos.append(msg % args if args else msg)

    def warning(self, msg, *args):
        self.warnings.append(msg % args if args else msg)


class TestBumpWandbRunStep(unittest.TestCase):
    def test_assigns_private_step_and_logs(self):
        run = _FakeWandbRun(21)
        logger = _FakeLogger()
        out = bump_wandb_run_step(run, 752, logger=logger)
        self.assertEqual(out, 753)
        self.assertEqual(run._step, 753)
        self.assertEqual(len(logger.infos), 1)
        self.assertIn("bumped _step from 21 to 753", logger.infos[0])
        self.assertEqual(logger.warnings, [])

    def test_noop_when_already_ahead(self):
        run = _FakeWandbRun(800)
        logger = _FakeLogger()
        out = bump_wandb_run_step(run, 752, logger=logger)
        self.assertEqual(out, 800)
        self.assertEqual(run._step, 800)
        self.assertEqual(logger.infos, [])

    def test_skips_and_warns_when_history_unknown(self):
        run = _FakeWandbRun(21)
        logger = _FakeLogger()
        out = bump_wandb_run_step(run, None, logger=logger)
        self.assertEqual(out, 21)
        self.assertEqual(run._step, 21)
        self.assertEqual(len(logger.warnings), 1)
        self.assertIn("skipping _step bump", logger.warnings[0])


class TestFetchLastHistoryStep(unittest.TestCase):
    def test_returns_int_from_api_run(self):
        api = MagicMock()
        api.run.return_value = SimpleNamespace(lastHistoryStep=752)
        self.assertEqual(
            fetch_last_history_step("flair3d_multi", "es6alp7s", api=api),
            752,
        )
        api.run.assert_called_once_with("flair3d_multi/es6alp7s")

    def test_includes_entity_in_path(self):
        api = MagicMock()
        api.run.return_value = SimpleNamespace(lastHistoryStep=1)
        fetch_last_history_step("proj", "rid", entity="ent", api=api)
        api.run.assert_called_once_with("ent/proj/rid")

    def test_returns_none_on_api_failure(self):
        api = MagicMock()
        api.run.side_effect = RuntimeError("offline")
        self.assertIsNone(fetch_last_history_step("proj", "rid", api=api))


class TestBumpOnResume(unittest.TestCase):
    def test_wires_fetch_then_bump(self):
        api = MagicMock()
        api.run.return_value = SimpleNamespace(lastHistoryStep=752)
        run = _FakeWandbRun(21)
        logger = _FakeLogger()
        out = bump_wandb_step_on_resume(
            run,
            project="flair3d_multi",
            run_id="es6alp7s",
            logger=logger,
            api=api,
        )
        self.assertEqual(out, 753)
        self.assertEqual(run._step, 753)

    def test_offline_run_skips_api_call(self):
        # Regression test: on a cluster compute node without internet egress
        # (e.g. Jean Zay), wandb.Api() blocks in an unbounded retry loop on
        # ConnectionError instead of raising, so an offline run must never
        # reach the api.run(...) call at all when no local step is known.
        api = MagicMock()
        run = _FakeWandbRun(21)
        run.settings = SimpleNamespace(mode="offline")
        logger = _FakeLogger()
        out = bump_wandb_step_on_resume(
            run,
            project="flair3d_multi",
            run_id="es6alp7s",
            logger=logger,
            api=api,
        )
        self.assertEqual(out, 21)
        self.assertEqual(run._step, 21)
        api.run.assert_not_called()

    def test_local_last_step_bumps_without_touching_api(self):
        # The common offline-resume path: CheckpointSaver already persisted
        # the step locally, so the bump must use it directly and never call
        # wandb.Api() at all (online or offline).
        api = MagicMock()
        run = _FakeWandbRun(21)
        run.settings = SimpleNamespace(mode="offline")
        logger = _FakeLogger()
        out = bump_wandb_step_on_resume(
            run,
            project="flair3d_multi",
            run_id="es6alp7s",
            logger=logger,
            api=api,
            local_last_step=752,
        )
        self.assertEqual(out, 753)
        self.assertEqual(run._step, 753)
        api.run.assert_not_called()


class TestLocalLastStep(unittest.TestCase):
    def test_round_trips_through_sidecar_file(self):
        with tempfile.TemporaryDirectory() as save_path:
            self.assertIsNone(read_local_last_step(save_path))
            write_local_last_step(save_path, 752)
            self.assertEqual(read_local_last_step(save_path), 752)
            write_local_last_step(save_path, 900)
            self.assertEqual(read_local_last_step(save_path), 900)

    def test_missing_directory_does_not_raise(self):
        write_local_last_step("/nonexistent/dir/for/sure", 1)  # must not raise

    def test_corrupted_file_returns_none(self):
        with tempfile.TemporaryDirectory() as save_path:
            import os

            with open(os.path.join(save_path, "wandb_last_step.txt"), "w") as f:
                f.write("not-an-int")
            self.assertIsNone(read_local_last_step(save_path))


class TestResumeStepCollisionScenario(unittest.TestCase):
    """Replay of LPT-B multi 1.2 (W&B run es6alp7s): job 1 vals at
    epochs 5..25 lived at _step 51, 102, ...; job 2 train logs of
    epochs 78, 83, ... reused those steps and W&B merged the rows.
    """

    JOB1_VAL_STEPS = {5: 51, 10: 102, 15: 153, 20: 204, 25: 255, 70: 714}
    LAST_HISTORY_STEP = 752
    JOB2_LOCAL_STEP_AFTER_INIT = 21

    def test_bumped_step_is_past_every_job1_history_step(self):
        new_step = compute_resume_step(
            self.JOB2_LOCAL_STEP_AFTER_INIT, self.LAST_HISTORY_STEP
        )
        self.assertEqual(new_step, 753)
        for epoch, old_step in self.JOB1_VAL_STEPS.items():
            self.assertGreater(
                new_step,
                old_step,
                msg=f"epoch {epoch} job1 _step {old_step} would still collide",
            )

    def test_unbumped_job2_train_epochs_collide_with_job1_vals(self):
        # Without the bump, job 2's first wandb.log calls reuse low _steps.
        colliding = {
            78: 51,
            83: 102,
            88: 153,
            93: 204,
            98: 255,
        }
        for train_epoch, job1_val_step in colliding.items():
            self.assertIn(job1_val_step, self.JOB1_VAL_STEPS.values())
            self.assertGreater(self.LAST_HISTORY_STEP, job1_val_step)
            # The train-only epochs are not multiples of eval_every=5.
            self.assertNotEqual(train_epoch % 5, 0)


if __name__ == "__main__":
    unittest.main()
