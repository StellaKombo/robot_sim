"""Use this checkout's model and policy when running from another directory."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ["ROBOT_SIM_ROOT_DIR"] = str(ROOT)
sys.path.insert(0, str(ROOT))
