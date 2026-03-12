#!/usr/bin/env python3
"""Generate multiple random offline street demos and stitch them into one figure."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


SCRIPT_PATH = Path(__file__).resolve()
HYBRID_ASTAR_ROOT = SCRIPT_PATH.parent.parent
OFFLINE_DEMO_SCRIPT = SCRIPT_PATH.parent / "offline_street_guided_astar_demo.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-generate random street guided A* demos and stitch them.")
    parser.add_argument("--count", type=int, default=4, help="number of random cases to generate")
    parser.add_argument("--cols", type=int, default=2, help="montage columns")
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="train")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HYBRID_ASTAR_ROOT / "offline_results" / "street_guided_demo",
    )
    parser.add_argument("--case-prefix", type=str, default="street_demo_batch")
    parser.add_argument("--python", type=str, default=sys.executable, help="python used for single-case script")
    return parser.parse_args()


def run_case(args: argparse.Namespace, seed: int, case_name: str) -> Path:
    cmd = [
        args.python,
        str(OFFLINE_DEMO_SCRIPT),
        "--split",
        args.split,
        "--map-index",
        "-1",
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--case-name",
        case_name,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return args.output_dir / case_name


def make_montage(case_dirs: list[Path], cols: int, out_path: Path) -> None:
    rows = math.ceil(len(case_dirs) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6.2, rows * 5.2), facecolor="#e6e6e6")
    if rows == 1 and cols == 1:
        axes_list = [axes]
    else:
        axes_list = list(axes.flat)

    for ax in axes_list:
        ax.set_visible(False)

    for ax, case_dir in zip(axes_list, case_dirs):
        ax.set_visible(True)
        image = mpimg.imread(case_dir / "offline_demo.png")
        ax.imshow(image)
        ax.axis("off")
        meta = json.loads((case_dir / "meta.json").read_text(encoding="utf-8"))
        ax.set_title(
            f"{case_dir.name}\nmap={meta['map_index']} start={tuple(meta['start_xy'])} goal={tuple(meta['goal_xy'])}",
            fontsize=10,
            pad=8,
        )

    fig.tight_layout(pad=1.0, w_pad=0.8, h_pad=1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    args = parse_args()
    case_dirs: list[Path] = []
    for index in range(args.count):
        seed = args.base_seed + index
        case_name = f"{args.case_prefix}_{index:02d}"
        case_dirs.append(run_case(args, seed, case_name))

    montage_path = args.output_dir / f"{args.case_prefix}_montage.png"
    make_montage(case_dirs, args.cols, montage_path)
    print(f"saved_montage={montage_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
