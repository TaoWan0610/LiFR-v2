from pathlib import Path
import sys

# Support direct invocation as well as the repository launcher.
for root in (Path(__file__).resolve().parents[1], Path(__file__).resolve().parents[3]):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from moge.train.trainer import main

if __name__ == '__main__':
    main()
