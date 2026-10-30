"""
Model Builder

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import copy
from pointcept.utils.registry import Registry

MODELS = Registry("models")


def build_model(cfg):
    """Build models."""
    return MODELS.build(copy.deepcopy(cfg))
