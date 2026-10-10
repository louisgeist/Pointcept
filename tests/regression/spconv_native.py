"""Force every spconv layer to ConvAlgo.Native.

The default auto-tuned algorithm is picked per process, so the same forward pass can differ by ~1e-2
between two runs, and on hecate (spconv-cu118 2.3.8) the tuner crashes as soon as a second SubMConv3d
shape is tuned. Native is deterministic in its algorithm choice and numerically correct, only slower.
Patched here rather than through a config hook so the harness does not depend on code the cleanup may
remove.
"""

import spconv.pytorch.conv as spconv_conv
from spconv.pytorch.core import ConvAlgo

_init = spconv_conv.SparseConvolution.__init__


def _native_init(self, *args, **kwargs):
    kwargs["algo"] = ConvAlgo.Native
    _init(self, *args, **kwargs)


spconv_conv.SparseConvolution.__init__ = _native_init
