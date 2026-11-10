"""Mixed-precision context for evaluation forwards (cfg.enable_amp_eval)."""

import contextlib

import torch

AMP_DTYPE = dict(float16=torch.float16, bfloat16=torch.bfloat16)


def eval_autocast(cfg):
    """CUDA autocast with cfg.amp_dtype when cfg.enable_amp_eval, else a no-op."""
    if not getattr(cfg, "enable_amp_eval", False):
        return contextlib.nullcontext()
    return torch.autocast(
        "cuda", dtype=AMP_DTYPE[getattr(cfg, "amp_dtype", "float16")]
    )
