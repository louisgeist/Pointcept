"""tools/train.py with every spconv layer forced to ConvAlgo.Native (see spconv_native.py)."""

import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spconv_native  # noqa: E402,F401

sys.argv = ["tools/train.py"] + sys.argv[1:]
runpy.run_path("tools/train.py", run_name="__main__")
