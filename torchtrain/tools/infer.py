#!/usr/bin/env python
"""PaddleDetection-style alias for `tools/predict.py`. See `tools/eval.py`."""

import os
import runpy
import sys

_TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.argv[0] = os.path.join(_TOOLS, "predict.py")
runpy.run_path(sys.argv[0], run_name="__main__")
