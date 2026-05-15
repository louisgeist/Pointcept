"""Progress helpers for evaluation loops (rank-0 tqdm bar)."""

from typing import Any, Dict, Mapping, Optional

import pointcept.utils.comm as comm


def _format_postfix_items(postfix: Mapping[str, Any]) -> Dict[str, str]:
    """Build short string values for tqdm postfix."""
    out: Dict[str, str] = {}
    for key, raw in postfix.items():
        if isinstance(raw, float):
            out[str(key)] = f"{raw:.4f}"
        elif isinstance(raw, bool):
            out[str(key)] = "Yes" if raw else "No"
        elif raw is None:
            out[str(key)] = ""
        else:
            out[str(key)] = str(raw)
    return out


class EvaluationProgressBar:
    """Rank-0 tqdm bar for validation/test loops. Other ranks: no output."""

    __slots__ = ("_total", "_desc", "_pbar", "_entered")

    def __init__(
        self,
        total: int,
        *,
        desc: str,
    ):
        self._total = max(int(total), 1)
        self._desc = desc
        self._pbar = None
        self._entered = False

    def __enter__(self) -> "EvaluationProgressBar":
        self._entered = True
        if not comm.is_main_process():
            return self
        from tqdm.auto import tqdm

        self._pbar = tqdm(
            total=self._total,
            desc=self._desc,
            dynamic_ncols=True,
            mininterval=0.25,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Optional[bool]:
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None
        self._entered = False
        return None

    def step(self, **postfix: Any) -> None:
        """Advance by one iteration; optional key-value pairs shown as tqdm postfix."""
        if not self._entered:
            raise RuntimeError(
                "EvaluationProgressBar.step() called outside context manager."
            )
        if not comm.is_main_process():
            return
        if self._pbar is None:
            return
        if postfix:
            self._pbar.set_postfix(**_format_postfix_items(postfix), refresh=False)
        self._pbar.update(1)
