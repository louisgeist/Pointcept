"""
Default Hook

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import pointcept.utils.comm as comm
import weakref
from .builder import HOOKS


class HookBase:
    """
    Base class for hooks that can be registered with :class:`TrainerBase`.
    """

    trainer = None  # A weak reference to the trainer object.

    def before_train(self):
        pass

    def before_epoch(self):
        pass

    def before_step(self):
        pass

    def after_step(self):
        pass

    def after_epoch(self):
        pass

    def after_train(self):
        pass

    def state_dict(self):
        """Optional hook-local state to persist across resume.

        Overridden by hooks that track running state across the whole training run
        in instance attributes (e.g. a per-task "best metric so far" used only for
        W&B/TensorBoard display) rather than in trainer-level state that
        CheckpointSaver/CheckpointLoader already checkpoint (``trainer.best_metric_value``).
        Without this, such trackers silently reset to their initial sentinel on every
        resume, making a "best so far" curve dip after a resume even though it should be
        a monotonic running max. Return {} (default) for hooks with nothing to persist.
        """
        return {}

    def load_state_dict(self, state):
        """Restore hook-local state saved via :meth:`state_dict`. No-op by default."""
        pass

    def should_evaluate(self):
        cfg = self.trainer.cfg
        if not cfg.evaluate:
            return False

        # Classic mode 
        # For backward compatibility, eval_every to 1,
        # so that it is eval_epoch that determines the validation frequency.
        if getattr(cfg, "eval_every", None) is None:
            return True
        
        # Iter-limited mode: evaluate every N epochs
        eval_every = getattr(cfg, "eval_every", 1)
        epoch = self.trainer.epoch + 1
        max_epoch = self.trainer.max_epoch
        return epoch % eval_every == 0 or epoch >= max_epoch


@HOOKS.register_module()
class ModelHook(HookBase):
    def before_train(self):
        if comm.get_world_size() > 1 and isinstance(
            self.trainer.model.module, HookBase
        ):
            self.model = weakref.proxy(self.trainer.model.module)
        elif isinstance(self.trainer.model, HookBase):
            self.model = weakref.proxy(self.trainer.model)
        else:
            self.model = HookBase()
        self.model.trainer = self.trainer
        self.model.before_train()

    def before_epoch(self):
        self.model.before_epoch()

    def before_step(self):
        self.model.before_step()

    def after_step(self):
        self.model.after_step()

    def after_epoch(self):
        self.model.after_epoch()

    def after_train(self):
        self.model.after_train()
