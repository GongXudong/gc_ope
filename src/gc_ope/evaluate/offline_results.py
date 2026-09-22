"""输出、独占锁和续跑校验；已完成的 checkpoint 始终只有一行。"""

from contextlib import contextmanager
from pathlib import Path
import fcntl
import hashlib
import json
import os
import pandas as pd
import numpy as np


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def result_lock(directory):
    """内核文件锁会在异常退出时释放，不把残留 PID 文件误认为活跃任务。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"已有运行占用输出目录：{directory}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def verify_manifest(directory, manifest):
    path = Path(directory) / "config.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("续跑配置、输入数据或代码已变化，请使用新的输出目录")
    else:
        if list(Path(directory).glob("*.csv")):
            raise ValueError("发现没有配置清单的旧 CSV，拒绝混用结果")
        atomic_json(path, manifest)


def read_rows(path):
    path = Path(path)
    if not path.exists():
        return {}
    frame = pd.read_csv(path, keep_default_na=False)
    if frame["checkpoint"].duplicated().any():
        raise ValueError(f"结果存在重复 checkpoint：{path}")
    return {int(row["checkpoint"]): row for row in frame.to_dict("records")}


def save_row(path, row):
    """重试覆盖对应 checkpoint，先写临时文件再替换，避免中断留下半行。"""
    path = Path(path)
    rows = read_rows(path)
    rows[row["checkpoint"]] = row
    temporary = path.with_suffix(".csv.tmp")
    pd.DataFrame([rows[key] for key in sorted(rows)]).to_csv(temporary, index=False)
    os.replace(temporary, path)


def completed(row):
    if str(row["status"]).startswith("skipped:"):
        return True
    if row["status"] != "ok":
        return False
    return bool(np.isfinite([float(row["kl"]), float(row["kl_seed_std"])]).all())


def audit_result(path, expected):
    rows = read_rows(path)
    missing = sorted(set(expected) - rows.keys())
    extra = sorted(rows.keys() - set(expected))
    invalid = [step for step, row in rows.items() if not completed(row)]
    return {"missing": missing, "extra": extra, "invalid": invalid,
            "ok": sum(row["status"] == "ok" for row in rows.values()),
            "skipped": sum(str(row["status"]).startswith("skipped:") for row in rows.values())}
