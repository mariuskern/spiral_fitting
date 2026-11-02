"""Export fiber direction samples from an existing HTTP(S) prediction manifest."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
EXTRACTOR = ROOT / "villa_8aa507c" / "fiber_direction_samples.py"


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    return subprocess.run([sys.executable, str(EXTRACTOR), *arguments]).returncode


if __name__ == "__main__":
    raise SystemExit(main())
