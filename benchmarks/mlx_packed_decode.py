from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torque_mlx.benchmarking import run_mlx_packed_decode_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--bit-width", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run_mlx_packed_decode_benchmark(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
