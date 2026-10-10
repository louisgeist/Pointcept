"""tools/train.py with every spconv layer forced to ConvAlgo.Native.

hecate's spconv-cu118 2.3.8 auto-tuner crashes as soon as a second SubMConv3d shape is tuned in the same
process; Native is numerically correct, only slower. Patched here rather than through a config hook so the
harness does not depend on code the cleanup may remove.
"""

import runpy
import sys

import spconv.pytorch.conv as spconv_conv
from spconv.pytorch.core import ConvAlgo

_init = spconv_conv.SparseConvolution.__init__


def _native_init(self, *args, **kwargs):
    kwargs["algo"] = ConvAlgo.Native
    _init(self, *args, **kwargs)


spconv_conv.SparseConvolution.__init__ = _native_init

sys.argv = ["tools/train.py"] + sys.argv[1:]
runpy.run_path("tools/train.py", run_name="__main__")
