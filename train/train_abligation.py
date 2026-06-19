"""Compatibility wrapper for the historical misspelled ablation entrypoint."""

import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from train_ablation import main


if __name__ == "__main__":
    main()
