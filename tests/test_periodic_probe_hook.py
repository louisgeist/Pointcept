"""PeriodicProbeHook: command formatting, one launch per checkpoint (also after a resume),
missing checkpoint, the summary of the probes' metrics.json, and the command taken from the
config entry probe_command.

Run with: PYTHONPATH=. pytest tests/test_periodic_probe_hook.py
"""

import csv
import json
import os
import tempfile
import types
import unittest

from pointcept.engines.hooks.misc import PeriodicProbeHook
from pointcept.utils.config import ConfigDict

# Writes the arguments and a metrics.json, as a probe run with MetricsJsonWriter would.
COMMAND = (
    "echo {weight} {epoch} {iter} > {probe_dir}/args.txt && "
    'echo \'{{"metric_name": "mIoU", "best_metric_value": 0.5}}\' > {probe_dir}/metrics.json'
)


class _Logger:
    def __init__(self):
        self.warnings = []

    def info(self, *args):
        pass

    def warning(self, msg, *args):
        self.warnings.append(msg)


def _trainer(save_path, epoch, probe_command=None):
    return types.SimpleNamespace(
        cfg=ConfigDict(save_path=save_path, probe_command=probe_command),
        epoch=epoch - 1,
        logger=_Logger(),
        train_loader=[None] * 7,
    )


class TestPeriodicProbeHook(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.save_path = self._tmp.name
        os.makedirs(os.path.join(self.save_path, "model"))

    def tearDown(self):
        self._tmp.cleanup()

    def _checkpoint(self, epoch):
        path = os.path.join(self.save_path, "model", f"epoch_{epoch}.pth")
        open(path, "w").close()
        return path

    def _run(self, hook, epoch, probe_command=None):
        hook.trainer = _trainer(self.save_path, epoch, probe_command)
        hook.before_train()
        hook.after_epoch()
        for process in hook._processes:
            process.wait()

    def test_launch_once_and_summary(self):
        weight = self._checkpoint(2)
        hook = PeriodicProbeHook(command=COMMAND, save_freq=2, iter_per_epoch=1000)
        self._run(hook, 1)  # not a probe epoch
        self.assertFalse(
            os.path.isdir(os.path.join(self.save_path, "probes", "epoch_1"))
        )
        self._run(hook, 2)
        probe_dir = os.path.join(self.save_path, "probes", "epoch_2")
        with open(os.path.join(probe_dir, "args.txt")) as f:
            self.assertEqual(f.read().split(), [os.path.abspath(weight), "2", "2000"])

        # A resumed run (new hook, same save_path) does not launch the probe again.
        os.remove(os.path.join(probe_dir, "args.txt"))
        self._run(PeriodicProbeHook(command=COMMAND, save_freq=2), 2)
        self.assertFalse(os.path.isfile(os.path.join(probe_dir, "args.txt")))

        hook.after_train()
        with open(os.path.join(self.save_path, "probes", "summary.csv")) as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(
            rows, [dict(epoch="2", metric_name="mIoU", best_metric_value="0.5")]
        )
        with open(os.path.join(self.save_path, "probes", "launched.json")) as f:
            self.assertEqual(list(json.load(f)), ["epoch_2"])

    def test_iter_defaults_to_loader_length(self):
        self._checkpoint(1)
        self._run(PeriodicProbeHook(command=COMMAND, save_freq=1), 1)
        with open(os.path.join(self.save_path, "probes", "epoch_1", "args.txt")) as f:
            self.assertEqual(f.read().split()[2], "7")

    def test_missing_checkpoint_warns(self):
        hook = PeriodicProbeHook(command=COMMAND, save_freq=1)
        self._run(hook, 1)
        self.assertEqual(len(hook.trainer.logger.warnings), 1)
        self.assertFalse(
            os.path.isfile(os.path.join(self.save_path, "probes", "launched.json"))
        )

    def test_command_from_config(self):
        self._checkpoint(1)
        self._run(PeriodicProbeHook(save_freq=1), 1, probe_command=COMMAND)
        self.assertTrue(
            os.path.isfile(
                os.path.join(self.save_path, "probes", "epoch_1", "args.txt")
            )
        )

    def test_disabled_without_command(self):
        self._checkpoint(1)
        self._run(PeriodicProbeHook(command=None, save_freq=1), 1)
        self.assertFalse(os.path.isdir(os.path.join(self.save_path, "probes")))


if __name__ == "__main__":
    unittest.main()
