"""MoGe depth package and native command-line entry points."""
from pathlib import Path
import sys

_repository_root = str(Path(__file__).resolve().parents[3])
if _repository_root not in sys.path:
    sys.path.insert(0, _repository_root)
