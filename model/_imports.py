"""Resolve the native task packages relative to this repository."""
from pathlib import Path
import sys


def use_task_packages(task):
    root = Path(__file__).resolve().parent
    paths = [root.parent, root / task]
    if task == 'multi_task':
        # Multi-task distillation also uses the semantic dataset adapter.
        paths.append(root / 'seg')
    for path in paths:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
