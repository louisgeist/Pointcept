"""Guard: CheckpointLoader must crash when a requested checkpoint is missing.

Scratch training (weight=None, resume=False) stays a no-op. A set but missing
path, or resume=True with no weight, used to log and train from scratch.

Run with: PYTHONPATH=./ pytest tests/test_checkpoint_loader.py
"""

import unittest
from unittest.mock import MagicMock

from pointcept.engines.hooks.misc import CheckpointLoader


def _fake_trainer(weight=None, resume=False):
    trainer = MagicMock()
    trainer.cfg.weight = weight
    trainer.cfg.resume = resume
    return trainer


class TestCheckpointLoaderMissingWeight(unittest.TestCase):
    def test_scratch_with_no_weight_is_noop(self):
        hook = CheckpointLoader()
        hook.trainer = _fake_trainer(weight=None, resume=False)
        hook.before_train()  # must not raise
        hook.trainer.model.load_state_dict.assert_not_called()

    def test_missing_file_raises(self):
        hook = CheckpointLoader()
        hook.trainer = _fake_trainer(
            weight="/this/path/does/not/exist.pth", resume=False
        )
        with self.assertRaises(RuntimeError) as ctx:
            hook.before_train()
        self.assertIn("No checkpoint found", str(ctx.exception))
        hook.trainer.model.load_state_dict.assert_not_called()

    def test_resume_without_weight_raises(self):
        hook = CheckpointLoader()
        hook.trainer = _fake_trainer(weight=None, resume=True)
        with self.assertRaises(RuntimeError) as ctx:
            hook.before_train()
        self.assertIn("resume=True", str(ctx.exception))
        hook.trainer.model.load_state_dict.assert_not_called()
