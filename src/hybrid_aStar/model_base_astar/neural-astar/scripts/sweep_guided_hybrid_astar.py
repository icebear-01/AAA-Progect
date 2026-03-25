"""Sweep lambda_guidance for guided Hybrid A* and summarize metrics.

Usage example:
  PYTHONPATH=src python -u scripts/sweep_guided_hybrid_astar.py \
    --lambdas 0.2 0.4 0.6 0.8 1.0 \
    --out-csv outputs/lambda_sweep/street_seed123.csv \
    -- \
    --occ-npz planning-datasets/data/street/mixed_064_moore_c16.npz \
    --random-map-index --random-start-goal --occ-semantics auto \
    --num-problems 20 --seed 123 \
    --start-yaw-deg 0 --goal-yaw-deg 0 \
    --ckpt outputs/model_guidance_street/best.pt --device cuda \
    --max-expansions 20000
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


BASELINE_RE = re.compile(
    r"baseline:\s*success_rate=([0-9.]+),\s*expanded_nodes=([0-9.]+),\s*runtime_ms=([0-9.]+)"
)
GUIDED_RE = re.compile(
    r"guided\(lambda=([0-9.]+)\):\s*success_rate=([0-9.]+),\s*expanded_nodes=([0-9.]+),\s*runtime_ms=([0-9.]+)"
)
COMMON_LEN_RE = re.compile(
    r"path_length\(common_success_only\):\s*count=([0-9]+),\s*baseline=([0-9.]+),\s*guided=([0-9.]+)"
)


@dataclass
class SweepRow:
    lam: float
    baseline_success: float
    baseline_expanded: float
    baseline_runtime_ms: float
    guided_success: float
    guided_expanded: float
    guided_runtime_ms: float
    common_count: int
    baseline_common_path: float
    guided_common_path: float
    return_code: int = 0
    log_path: Optional[Path] = None

    @property
    def delta_success(self) -> float:
        return self.guided_success - self.baseline_success

    @property
    def delta_expanded(self) -> float:
        return self.guided_expanded - self.baseline_expanded

    @property
    def delta_runtime_ms(self) -> float:
        return self.guided_runtime_ms - self.baseline_runtime_ms

    @property
    def delta_common_path(self) -> float:
        return self.guided_common_path - self.baseline_common_path


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(description="Sweep lambda_guidance and summarize eval metrics")
    parser.add_argument(
        "--eval-script",
        type=Path,
        default=Path("scripts/eval_guided_hybrid_astar_demo.py"),
        help="Eval script path",
    )
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        required=True,
        help="Lambda values to sweep",
    )
    parser.add_argument(
        "--python-exe",
        type=str,
        default=sys.executable,
        help="Python executable used to run eval script",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("outputs/lambda_sweep/sweep.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--save-logs",
        action="store_true",
        help="Save full stdout/stderr for each lambda",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("outputs/lambda_sweep/logs"),
        help="Log directory used when --save-logs is set",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first eval failure",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=0,
        help="Timeout seconds per lambda (0 means no timeout)",
    )
    args, extra = parser.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    return args, extra


def _strip_lambda_args(argv: Sequence[str]) -> List[str]:
    out: List[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--lambda-guidance":
            i += 2
            continue
        if token.startswith("--lambda-guidance="):
            i += 1
            continue
        out.append(token)
        i += 1
    return out


def _parse_summary(text: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[int, float, float]]:
    m_b = BASELINE_RE.search(text)
    m_g = GUIDED_RE.search(text)
    m_l = COMMON_LEN_RE.search(text)
    if m_b is None or m_g is None or m_l is None:
        raise ValueError("Failed to parse summary lines from eval output")

    baseline = (
        float(m_b.group(1)),
        float(m_b.group(2)),
        float(m_b.group(3)),
    )
    guided = (
        float(m_g.group(2)),
        float(m_g.group(3)),
        float(m_g.group(4)),
    )
    common = (
        int(m_l.group(1)),
        float(m_l.group(2)),
        float(m_l.group(3)),
    )
    return baseline, guided, common


def _fmt(v: float, nd: int = 3) -> str:
    return f"{v:.{nd}f}"


def _print_table(rows: List[SweepRow]) -> None:
    headers = [
        "lambda",
        "b_succ",
        "g_succ",
        "d_succ",
        "b_exp",
        "g_exp",
        "d_exp",
        "b_ms",
        "g_ms",
        "d_ms",
        "n_common",
        "b_len",
        "g_len",
        "d_len",
    ]
    print("\n=== Lambda Sweep Table ===")
    print(" | ".join(headers))
    print("-" * 160)
    for r in rows:
        fields = [
            _fmt(r.lam, 2),
            _fmt(r.baseline_success),
            _fmt(r.guided_success),
            _fmt(r.delta_success),
            _fmt(r.baseline_expanded, 1),
            _fmt(r.guided_expanded, 1),
            _fmt(r.delta_expanded, 1),
            _fmt(r.baseline_runtime_ms),
            _fmt(r.guided_runtime_ms),
            _fmt(r.delta_runtime_ms),
            str(r.common_count),
            _fmt(r.baseline_common_path),
            _fmt(r.guided_common_path),
            _fmt(r.delta_common_path),
        ]
        print(" | ".join(fields))


def _write_csv(rows: List[SweepRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "lambda",
                "baseline_success",
                "guided_success",
                "delta_success",
                "baseline_expanded",
                "guided_expanded",
                "delta_expanded",
                "baseline_runtime_ms",
                "guided_runtime_ms",
                "delta_runtime_ms",
                "common_count",
                "baseline_common_path",
                "guided_common_path",
                "delta_common_path",
                "return_code",
                "log_path",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.lam,
                    r.baseline_success,
                    r.guided_success,
                    r.delta_success,
                    r.baseline_expanded,
                    r.guided_expanded,
                    r.delta_expanded,
                    r.baseline_runtime_ms,
                    r.guided_runtime_ms,
                    r.delta_runtime_ms,
                    r.common_count,
                    r.baseline_common_path,
                    r.guided_common_path,
                    r.delta_common_path,
                    r.return_code,
                    str(r.log_path) if r.log_path is not None else "",
                ]
            )


def _choose_best(rows: List[SweepRow]) -> SweepRow:
    # Priority:
    # 1) higher guided success
    # 2) lower guided expanded nodes
    # 3) lower guided runtime
    # 4) lower guided common path
    return min(
        rows,
        key=lambda r: (
            -r.guided_success,
            r.guided_expanded,
            r.guided_runtime_ms,
            r.guided_common_path,
        ),
    )


def main() -> None:
    args, eval_args = parse_args()
    eval_args = _strip_lambda_args(eval_args)

    if not args.eval_script.exists():
        raise FileNotFoundError(f"eval script not found: {args.eval_script}")

    if args.save_logs:
        args.logs_dir.mkdir(parents=True, exist_ok=True)

    rows: List[SweepRow] = []
    for lam in args.lambdas:
        cmd = [
            args.python_exe,
            "-u",
            str(args.eval_script),
            *eval_args,
            "--lambda-guidance",
            str(float(lam)),
        ]
        print(f"\n[sweep] lambda={lam:.3f}")
        print("[cmd] " + " ".join(cmd))

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=(None if args.timeout_sec <= 0 else args.timeout_sec),
            )
            output = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
            log_path = None
            if args.save_logs:
                log_path = args.logs_dir / f"lambda_{lam:.3f}.log"
                log_path.write_text(output, encoding="utf-8")

            if proc.returncode != 0:
                print(f"[error] return_code={proc.returncode}")
                if args.fail_fast:
                    raise RuntimeError(f"eval failed at lambda={lam}")
                rows.append(
                    SweepRow(
                        lam=float(lam),
                        baseline_success=0.0,
                        baseline_expanded=0.0,
                        baseline_runtime_ms=0.0,
                        guided_success=0.0,
                        guided_expanded=0.0,
                        guided_runtime_ms=0.0,
                        common_count=0,
                        baseline_common_path=0.0,
                        guided_common_path=0.0,
                        return_code=proc.returncode,
                        log_path=log_path,
                    )
                )
                continue

            baseline, guided, common = _parse_summary(output)
            rows.append(
                SweepRow(
                    lam=float(lam),
                    baseline_success=baseline[0],
                    baseline_expanded=baseline[1],
                    baseline_runtime_ms=baseline[2],
                    guided_success=guided[0],
                    guided_expanded=guided[1],
                    guided_runtime_ms=guided[2],
                    common_count=common[0],
                    baseline_common_path=common[1],
                    guided_common_path=common[2],
                    return_code=proc.returncode,
                    log_path=log_path,
                )
            )
            print(
                "[ok] "
                f"guided_success={guided[0]:.3f}, "
                f"guided_expanded={guided[1]:.1f}, "
                f"guided_runtime_ms={guided[2]:.3f}"
            )

        except subprocess.TimeoutExpired:
            print(f"[timeout] lambda={lam:.3f}")
            if args.fail_fast:
                raise
            rows.append(
                SweepRow(
                    lam=float(lam),
                    baseline_success=0.0,
                    baseline_expanded=0.0,
                    baseline_runtime_ms=0.0,
                    guided_success=0.0,
                    guided_expanded=0.0,
                    guided_runtime_ms=0.0,
                    common_count=0,
                    baseline_common_path=0.0,
                    guided_common_path=0.0,
                    return_code=124,
                    log_path=None,
                )
            )

    if not rows:
        raise RuntimeError("No sweep result produced")

    rows_ok = [r for r in rows if r.return_code == 0]
    _print_table(rows)
    _write_csv(rows, args.out_csv)
    print(f"\nSaved CSV: {args.out_csv}")

    if rows_ok:
        best = _choose_best(rows_ok)
        print(
            "\nBest lambda (by success -> expanded -> runtime -> path): "
            f"{best.lam:.3f}"
        )
    else:
        print("\nNo successful runs to select best lambda.")


if __name__ == "__main__":
    main()

