"""从旧混合结果 CSV 中按 method/task/seed 拆分出单轨迹结果文件。

本脚本只做数据提取，不重新计算，不修改源 CSV。输出保持源文件的列和字段值
不变，并按 checkpoint 升序排序，便于后续绘图直接读取。

示例：
  conda run -n gc_ope python scripts/extract_method_seed_csv.py \
    --source "logs/replacement_experiment/all100/kde_gmm_hist_gaussian_kl copy.csv" \
    --task push --method gmm --seeds 1 2 3 4 5 \
    --output-dir logs/replacement_experiment/all100/method_per_seed \
    --overwrite
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "logs" / "replacement_experiment" / "all100" / "kde_gmm_hist_gaussian_kl copy.csv"
DEFAULT_OUTPUT_DIR = ROOT / "logs" / "replacement_experiment" / "all100" / "method_per_seed"

EXPECTED_COLUMNS = [
    "task",
    "seed",
    "checkpoint",
    "method",
    "kl",
    "kl_seed_std",
    "fixed_grid_kl",
    "historical_successes",
    "reference_successes",
    "status",
    "error",
    "job_time_s",
]
ALL100_CHECKPOINTS = [10000 + 10000 * i for i in range(99)] + [1000000]


def _read_rows(source: Path, task: str, method: str, seeds: list[int]) -> dict[int, list[dict[str, str]]]:
    if not source.is_file():
        raise FileNotFoundError(f"source CSV not found: {source}")

    with source.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(
                "source CSV columns do not match the expected 12-column schema: "
                f"{reader.fieldnames!r}"
            )

        grouped = {seed: [] for seed in seeds}
        seed_set = set(seeds)
        for row in reader:
            if row["task"] != task or row["method"] != method:
                continue
            try:
                seed = int(row["seed"])
                checkpoint = int(row["checkpoint"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid seed/checkpoint row: {row!r}") from exc
            if seed in seed_set:
                row["checkpoint"] = str(checkpoint)
                grouped[seed].append(row)

    for seed in seeds:
        grouped[seed].sort(key=lambda row: int(row["checkpoint"]))
    return grouped


def _validate_group(seed: int, rows: list[dict[str, str]]) -> None:
    checkpoints = [int(row["checkpoint"]) for row in rows]
    if checkpoints != ALL100_CHECKPOINTS:
        missing = sorted(set(ALL100_CHECKPOINTS) - set(checkpoints))
        extra = sorted(set(checkpoints) - set(ALL100_CHECKPOINTS))
        raise ValueError(
            f"seed {seed}: expected exactly 100 checkpoints 10000..1000000, "
            f"got {len(checkpoints)} rows; missing={missing}, extra={extra}"
        )
    if len(checkpoints) != len(set(checkpoints)):
        duplicated = sorted({c for c in checkpoints if checkpoints.count(c) > 1})
        raise ValueError(f"seed {seed}: duplicated checkpoints: {duplicated}")

    statuses = Counter(row["status"] for row in rows)
    invalid_status = set(statuses) - {"ok"} - {
        status for status in statuses if status.startswith("skipped:")
    }
    if invalid_status:
        raise ValueError(f"seed {seed}: contains non-final status rows: {sorted(invalid_status)}")


def _write_group(output_path: Path, rows: list[dict[str, str]], overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"output exists (use --overwrite to replace it): {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXPECTED_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--task", required=True, choices=["push", "slide"])
    parser.add_argument("--method", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的输出文件")
    args = parser.parse_args()

    if not args.seeds:
        parser.error("--seeds cannot be empty")
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds cannot contain duplicates")

    grouped = _read_rows(args.source, args.task, args.method, args.seeds)
    for seed in args.seeds:
        _validate_group(seed, grouped[seed])

    for seed in args.seeds:
        output_path = args.output_dir / f"{args.method}_{args.task}_seed{seed}.csv"
        _write_group(output_path, grouped[seed], overwrite=args.overwrite)

    print(f"source: {args.source}")
    print(f"extracted: task={args.task}, method={args.method}, seeds={args.seeds}")
    for seed in args.seeds:
        rows = grouped[seed]
        statuses = Counter(row["status"] for row in rows)
        ok = statuses.get("ok", 0)
        skipped = sum(count for status, count in statuses.items() if status.startswith("skipped:"))
        output_path = args.output_dir / f"{args.method}_{args.task}_seed{seed}.csv"
        print(f"  seed {seed}: {len(rows)} rows, ok={ok}, skipped={skipped}, error={len(rows)-ok-skipped}, file={output_path}")


if __name__ == "__main__":
    main()
