"""Where does a Volt training step spend GPU memory?

Builds the model and one training batch from a config, then runs a few AMP
forward/backward steps with hooks on the top-level modules (teacher, tokenizer,
transformer blocks, decoder, heads). For each module it prints the memory held
after its forward (activations kept for backward) and its transient peak; then
the peak of the backward pass and of the whole step.

    PYTHONPATH=. python scripts/volt/profile_memory.py CONFIG [--steps 3] [--options k=v ...]
"""

import argparse
from collections import OrderedDict

import torch

from pointcept.datasets import build_dataset, point_collate_fn
from pointcept.engines.defaults import default_config_parser
from pointcept.models import build_model
from pointcept.utils.amp import AMP_DTYPE

GiB = 1024**3


def watched_modules(model):
    out = OrderedDict()
    if getattr(model, "teacher", None) is not None:
        out["teacher"] = model.teacher
    bb = model.backbone
    for name in ("tokenizer", "rope", "blocks", "decoder"):
        if hasattr(bb, name):
            out[f"backbone.{name}"] = getattr(bb, name)
    for group in ("seg_heads", "reg_heads", "pixel_seg_heads", "distill_heads"):
        for k, m in getattr(model, group, {}).items():
            out[f"{group}.{k}"] = m
    return out


class Recorder:
    def __init__(self, modules):
        self.rows = OrderedDict((k, dict(calls=0, kept=0, peak=0)) for k in modules)
        self._before = {}
        for name, m in modules.items():
            m.register_forward_pre_hook(self._pre(name))
            m.register_forward_hook(self._post(name))

    def _pre(self, name):
        def f(*_):
            torch.cuda.synchronize()
            self._before[name] = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()

        return f

    def _post(self, name):
        def f(*_):
            torch.cuda.synchronize()
            before = self._before[name]
            r = self.rows[name]
            r["calls"] += 1
            r["kept"] += torch.cuda.memory_allocated() - before
            r["peak"] = max(r["peak"], torch.cuda.max_memory_allocated() - before)

        return f

    def reset(self):
        for r in self.rows.values():
            r.update(calls=0, kept=0, peak=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--options", nargs="*", default=[])
    args = ap.parse_args()
    opts = dict(o.split("=", 1) for o in args.options)
    opts = {k: eval(v) if v[:1].isdigit() or v in ("True", "False", "None") else v for k, v in opts.items()}
    cfg = default_config_parser(args.config, opts)

    model = build_model(cfg.model).cuda().train()
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.synchronize()
    print(f"params trainable {n_param / 1e6:.1f} M; after build+optimizer "
          f"{torch.cuda.memory_allocated() / GiB:.2f} GiB")

    ds = build_dataset(cfg.data.train)
    bs = cfg.batch_size_per_gpu if "batch_size_per_gpu" in cfg else cfg.batch_size
    rec = Recorder(watched_modules(model))
    amp_dtype = AMP_DTYPE[cfg.amp_dtype]
    n_tok = []
    if hasattr(model.backbone, "tokenizer"):
        model.backbone.tokenizer.register_forward_hook(lambda m, i, o: n_tok.append(int(o[0].shape[0])))

    for step in range(args.steps):
        batch = point_collate_fn([ds[(step * bs + i) % len(ds)] for i in range(bs)])
        batch = {k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        n_pts = int(batch["coord"].shape[0])
        rec.reset()
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        with torch.amp.autocast("cuda", enabled=cfg.enable_amp, dtype=amp_dtype):
            out = model(batch)
        torch.cuda.synchronize()
        after_fwd = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        scaler.scale(out["loss"]).backward()
        torch.cuda.synchronize()
        bwd_peak = torch.cuda.max_memory_allocated()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        del out
        torch.cuda.synchronize()
        print(f"\n=== step {step}: {bs} crops, {n_pts} points; base {base / GiB:.2f} GiB, "
              f"kept after forward {(after_fwd - base) / GiB:.2f} GiB, backward peak "
              f"{bwd_peak / GiB:.2f} GiB, reserved {torch.cuda.max_memory_reserved() / GiB:.2f} GiB")
        print(f"{'module':40s} {'calls':>5s} {'kept GiB':>9s} {'peak GiB':>9s}")
        for name, r in rec.rows.items():
            if r["calls"]:
                print(f"{name:40s} {r['calls']:5d} {r['kept'] / GiB:9.2f} {r['peak'] / GiB:9.2f}")
        if n_tok:
            print(f"tokens {n_tok[-1]} ({n_tok[-1] / bs:.0f}/crop)")


if __name__ == "__main__":
    main()
