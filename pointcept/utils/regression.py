"""Utilities for point-wise regression evaluation."""


def denorm_regression_prediction(pred, scale: float):
    """Map model output from normalized target space back to physical units."""
    if scale is None or scale == 1.0:
        return pred
    return pred / scale


def get_regression_target_scale(target_scales, target_key: str) -> float:
    if not target_scales:
        return 1.0
    return float(target_scales.get(target_key, 1.0))
