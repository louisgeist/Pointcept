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
        """Hook-local state saved in checkpoints and restored on resume.

        Override in hooks that keep run-long state in instance attributes (e.g. a
        per-task "best so far" metric used only for logging); trainer-level state such
        as ``trainer.best_metric_value`` is already checkpointed. Without it, such state
        resets on every resume. Returns {} by default.
        """
        return {}

    def load_state_dict(self, state):
        """Restore hook-local state saved via :meth:`state_dict`. No-op by default."""
        pass

    def should_evaluate(self):
        """Whether validation runs at the end of the current trainer epoch."""
        cfg = self.trainer.cfg
        if not cfg.evaluate:
            return False

        # Classic mode (eval_every is None): validate after every trainer epoch,
        # so the validation cadence is set by eval_epoch.
        if getattr(cfg, "eval_every", None) is None:
            return True
        
        # Iter-limited mode: every eval_every epochs, and after the last epoch.
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
