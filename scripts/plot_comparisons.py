"""Generate cross-condition comparison plots from resolved run artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rl_a2.plots import plot_comparisons  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="artifacts/full")
    args = parser.parse_args()
    paths = plot_comparisons(Path(args.output_root))
    print(f"Wrote {len(paths)} comparison plots under {Path(args.output_root) / 'plots'}")
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
