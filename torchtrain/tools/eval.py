#!/usr/bin/env python
"""PaddleDetection-style alias for `tools/val.py`.

The platform generates `tools/val.py` for torch jobs, but a user may paste a
command copied from a PaddleDetection workflow (`tools/eval.py -c cfg -o
weights=...`). Delegating keeps both spellings working.
"""

import os
import runpy
import sys

_TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.argv[0] = os.path.join(_TOOLS, "val.py")
runpy.run_path(sys.argv[0], run_name="__main__")
