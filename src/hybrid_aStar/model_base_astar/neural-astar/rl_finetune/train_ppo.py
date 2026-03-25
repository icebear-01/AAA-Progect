"""PPO fine-tuning scaffold (encoder-only updates).

This is intentionally a scaffold and not a full implementation.
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Route A PPO fine-tuning scaffold")
    p.add_argument("--enable", action="store_true", help="Acknowledge scaffold mode")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--steps", type=int, default=10000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.enable:
        print("Scaffold only. Re-run with --enable to acknowledge TODO implementation.")
        return

    print("TODO: integrate PPO loop to update GuidanceEncoder parameters only.")
    print("Suggested flow: sample map/task -> infer cost map -> guided planner -> reward -> PPO update.")
    raise NotImplementedError("RL fine-tuning is scaffold-only in this repository update.")


if __name__ == "__main__":
    main()
