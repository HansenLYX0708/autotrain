"""Compatibility CLI for the dataset preparer bundled with torchtrain."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "torchtrain"))
from torchtrain.localization_data import corners, label_line, prepare, read_project, validate, main

if __name__ == "__main__":
    main()
