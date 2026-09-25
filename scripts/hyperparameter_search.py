"""离线能力分布估计器的可复现超参数搜索入口。

本脚本只读取旧 fixed-eval CSV，不调用环境、不读取 replay buffer，也不接触
课程学习 wrapper。历史侧和当前 checkpoint reference 侧分别实例化同一种
evaluator；候选之间复用同一批历史抽样索引和 MC 索引。搜索结果写到独立的
``logs/hyperparameter_search`` 目录，支持按 candidate_id 断点续跑。

默认用法：
    conda run -n gc_ope python scripts/hyperparameter_search.py \
        --method gmm --task push --stage screening

正式复核必须显式给出筛选结果：
    conda run -n gc_ope python scripts/hyperparameter_search.py \
        --method gmm --task push --stage validation \
        --candidate-file logs/hyperparameter_search/push/gmm/screening/candidate_results.csv
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import re
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from scipy.special import logsumexp

from gc_ope.evaluate.evaluation_result_container import (
    EvaluationResultContainer,
    WeightedEvaluationResultContainer,
)
from gc_ope.evaluate.evaluator_fm import FlowMatchingDensityEvaluator
from gc_ope.evaluate.evaluator_gmm import GMMEvaluator
from gc_ope.evaluate.evaluator_nf import NormalizingFlowDensityEvaluator
from gc_ope.evaluate.evaluator_nn import GoalSuccessMLPClassifierEvaluator


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("gmm", "nn", "nf", "fm")
TASKS = ("push", "slide")
GOAL_COLUMNS = ("x", "y")
CHECKPOINT_RE = re.compile(r"^rl_model_(\d+)_steps_eval_res_on_fixed\.csv$")
TASK_CODES = {"push": 1, "slide": 2}


class SearchProtocolError(ValueError):
    """配置、数据或协议不满足搜索要求。"""


@dataclass(frozen=True)
class SourceFrame:
    """一个来源 checkpoint 的固定评估记录和确定性抽样结果。"""

    step: int
    path: Path
    frame: pd.DataFrame
    sampled_indices: np.ndarray


@dataclass
class TrajectoryData:
    task: str
    seed: int
    checkpoint: int
    reference: pd.DataFrame
    history: list[SourceFrame]
    mc_indices: list[np.ndarray]
    sample_hash: str
    input_paths: list[Path]
    phase: str


@dataclass
class FitResult:
    evaluator: Any
    warnings: list[str]
    diagnostics: dict[str, Any]


class FMEnsembleEvaluator:
    """FM 的独立成员等权密度混合，保留 evaluator 的离线公共接口。"""

    def __init__(self, *, n_members: int, member_kwargs: dict[str, Any], kappa: float):
        _validate_positive_integer("n_members", n_members)
        self.n_members = int(n_members)
        self.member_kwargs = dict(member_kwargs)
        self.eval_res_container = WeightedEvaluationResultContainer(discounted_factor=kappa)
        self.members_: list[FlowMatchingDensityEvaluator] = []
        self.fit_diagnostics_: dict[str, Any] = {}

    def fit_evaluator(self):
        self.members_ = []
        for index in range(self.n_members):
            member = FlowMatchingDensityEvaluator(
                evaluation_result_container_class=WeightedEvaluationResultContainer,
                evaluation_result_container_kwargs={"discounted_factor": self.eval_res_container.discounted_factor},
                random_state=int(self.member_kwargs["random_state"]) + index,
                **{key: value for key, value in self.member_kwargs.items() if key != "random_state"},
            )
            member.eval_res_container = self.eval_res_container
            member.fit_evaluator()
            self.members_.append(member)
        self.fit_diagnostics_ = {
            "n_members": self.n_members,
            "member_seeds": [int(self.member_kwargs["random_state"]) + index for index in range(self.n_members)],
            "members": [_json_safe(member.fit_diagnostics_) for member in self.members_],
            "density_scheme": "independent FM members with equal-weight raw-density mixture",
            "quality_warnings": [
                f"成员 {index}：{warning}"
                for index, member in enumerate(self.members_)
                for warning in member.fit_diagnostics_.get("quality_warnings", [])
            ],
        }
        return self

    def _require_fitted(self):
        if len(self.members_) != self.n_members:
            raise RuntimeError("FMEnsembleEvaluator requires fit_evaluator() first")

    def evaluate_grid(self, grid: np.ndarray, return_log_density: bool = False) -> np.ndarray:
        self._require_fitted()
        values = np.stack([
            member.evaluate_grid(grid, return_log_density=True) for member in self.members_
        ])
        mixed = logsumexp(values, axis=0) - math.log(self.n_members)
        return mixed if return_log_density else np.exp(np.clip(mixed, -745.0, 709.0))

    def sample(self, n_samples: int, random_state: int = 0) -> np.ndarray:
        self._require_fitted()
        rng = np.random.default_rng(random_state)
        members = rng.integers(self.n_members, size=int(n_samples))
        result = np.empty((int(n_samples), 2), dtype=float)
        for index, member in enumerate(self.members_):
            selected = members == index
            if np.any(selected):
                result[selected] = member.sample(int(np.sum(selected)), int(rng.integers(0, 2**31 - 1)))
        return result


RESULT_COLUMNS = [
    "candidate_id", "method", "stage", "status", "mean_kl", "median_kl",
    "worst_trajectory_kl", "early_kl", "middle_kl", "late_kl",
    "finite_result_ratio", "skipped_count", "fit_warnings", "runtime_s",
    "params_json", "error",
]

SUMMARY_COLUMNS = [
    "candidate_id", "method", "stage", "task", "seed", "checkpoint", "phase",
    "status", "mean_kl", "median_kl", "finite_result_ratio", "skipped_count",
    "history_rows", "history_successes", "reference_rows", "reference_successes",
    "fit_warnings", "runtime_s", "error", "sample_hash", "mc_index_hash",
]


def _canonical(value: Any) -> str:
    """把配置转成跨运行稳定的 JSON 字符串。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _json_safe(value: Any) -> Any:
    """把 numpy 标量、数组和路径转换成 JSON/YAML 可保存的值。"""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256(_canonical(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_indices(indices: Iterable[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for values in indices:
        array = np.asarray(values, dtype=np.int64)
        digest.update(array.tobytes())
    return digest.hexdigest()


def _read_fixed_csv(path: Path) -> pd.DataFrame:
    """按师兄 fixed-eval 格式读取并检查最小字段。"""
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {"x", "y", "termination"}
    missing = required.difference(frame.columns)
    if missing:
        raise SearchProtocolError(f"fixed CSV 缺少字段 {sorted(missing)}: {path}")
    goals = frame.loc[:, list(GOAL_COLUMNS)].to_numpy(dtype=float)
    if goals.ndim != 2 or len(goals) == 0 or not np.all(np.isfinite(goals)):
        raise SearchProtocolError(f"fixed CSV 的 x,y 为空或含非有限值: {path}")
    return frame.reset_index(drop=True)


def checkpoint_files(data_root: Path, task: str, seed: int) -> list[tuple[int, Path]]:
    """只发现 vanilla SAC 的 fixed-eval checkpoint CSV。"""
    if task not in TASKS:
        raise SearchProtocolError(f"只支持 push/slide，收到 {task}")
    directory = Path(data_root) / "checkpoints" / f"my_{task}" / "sac" / f"seed_{seed}"
    found: list[tuple[int, Path]] = []
    for path in directory.glob("rl_model_*_steps_eval_res_on_fixed.csv"):
        match = CHECKPOINT_RE.match(path.name)
        if match:
            found.append((int(match.group(1)), path))
    result = sorted(found)
    if not result:
        raise FileNotFoundError(f"没有找到 fixed-eval CSV: {directory}")
    return result


def _choose_checkpoints(
    available: list[int], stage_config: dict[str, Any], explicit: list[int] | None,
) -> list[int]:
    if explicit:
        selected = list(dict.fromkeys(int(value) for value in explicit))
    else:
        configured = stage_config.get("checkpoints")
        if configured in (None, "all"):
            selected = list(available)
        elif configured == "representative":
            positions = sorted(set([0, len(available) // 2, len(available) - 1]))
            selected = [available[position] for position in positions]
        else:
            selected = [int(value) for value in configured]
    missing = sorted(set(selected).difference(available))
    if missing:
        raise SearchProtocolError(f"配置中的 checkpoint 不存在: {missing}")
    if not selected:
        raise SearchProtocolError("没有可用于搜索的 checkpoint")
    return sorted(selected)


def _sample_history(
    data_root: Path,
    task: str,
    seed: int,
    checkpoint: int,
    samples_per_checkpoint: int,
    sampling_seed: int,
) -> tuple[pd.DataFrame, list[SourceFrame], list[Path]]:
    if samples_per_checkpoint <= 0:
        raise SearchProtocolError("samples_per_checkpoint 必须为正整数")
    files = [(step, path) for step, path in checkpoint_files(data_root, task, seed) if step <= checkpoint]
    if not files or files[-1][0] != checkpoint:
        raise SearchProtocolError(f"当前 checkpoint 不在 fixed-eval 文件集合中: {task}/seed_{seed}/{checkpoint}")
    history: list[SourceFrame] = []
    all_goals: list[pd.DataFrame] = []
    paths: list[Path] = []
    for step, path in files:
        frame = _read_fixed_csv(path)
        rng = np.random.default_rng(
            np.random.SeedSequence([int(sampling_seed), TASK_CODES[task], int(seed), int(step)])
        )
        indices = rng.integers(0, len(frame), size=samples_per_checkpoint, dtype=np.int64)
        sampled = frame.iloc[indices].reset_index(drop=True)
        history.append(SourceFrame(step, path, sampled, indices))
        all_goals.append(sampled.loc[:, list(GOAL_COLUMNS)])
        paths.append(path)
    # 返回合并后的副本，便于测试和审计；拟合仍保留来源 checkpoint 权重。
    return pd.concat(all_goals, ignore_index=True), history, paths


def _make_mc_indices(
    reference: pd.DataFrame, task: str, seed: int, checkpoint: int,
    mc_samples: int, mc_repeats: int, random_seed: int,
) -> list[np.ndarray]:
    if mc_samples <= 0 or mc_repeats <= 0:
        raise SearchProtocolError("mc_samples 和 mc_repeats 必须为正整数")
    rng = np.random.default_rng(
        np.random.SeedSequence([int(random_seed), TASK_CODES[task], int(seed), int(checkpoint), 991])
    )
    return [
        rng.integers(0, len(reference), size=mc_samples, dtype=np.int64)
        for _ in range(mc_repeats)
    ]


def load_trajectory(
    data_root: Path,
    task: str,
    seed: int,
    checkpoint: int,
    *,
    samples_per_checkpoint: int = 100,
    sampling_seed: int = 0,
    mc_samples: int = 256,
    mc_repeats: int = 1,
    mc_seed: int = 0,
    phase: str = "middle",
) -> TrajectoryData:
    """读取一个目标 checkpoint，并一次性固定其历史/MC 随机流。"""
    paths = dict(checkpoint_files(data_root, task, seed))
    if checkpoint not in paths:
        raise FileNotFoundError(paths.get(checkpoint, checkpoint))
    reference = _read_fixed_csv(paths[checkpoint])
    _, history, history_paths = _sample_history(
        data_root, task, seed, checkpoint, samples_per_checkpoint, sampling_seed
    )
    indices = _make_mc_indices(reference, task, seed, checkpoint, mc_samples, mc_repeats, mc_seed)
    sample_hash = _hash_indices([source.sampled_indices for source in history])
    return TrajectoryData(
        task=task,
        seed=seed,
        checkpoint=checkpoint,
        reference=reference,
        history=history,
        mc_indices=indices,
        sample_hash=sample_hash,
        input_paths=[*history_paths],
        phase=phase,
    )


def _history_arrays(
    history: list[SourceFrame], checkpoint: int, kappa: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    goals: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for source in history:
        goals.append(source.frame.loc[:, list(GOAL_COLUMNS)].to_numpy(dtype=float))
        labels.append((source.frame["termination"].to_numpy() == "reach target").astype(int))
        weight = float(kappa ** ((checkpoint - source.step) / 10000.0))
        weights.append(np.full(len(source.frame), weight, dtype=float))
    if not goals:
        raise SearchProtocolError("历史数据为空")
    return np.concatenate(goals), np.concatenate(labels), np.concatenate(weights)


def _validate_positive_integer(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
        raise SearchProtocolError(f"{name} 必须为正整数")


def validate_method_params(method: str, params: dict[str, Any]) -> dict[str, Any]:
    """方法独立地校验并规范化候选参数。"""
    if method not in METHODS:
        raise SearchProtocolError(f"未知 method: {method}")
    normalized = dict(params)
    if method == "gmm":
        _validate_positive_integer("n_components", normalized.get("n_components"))
        _validate_positive_integer("max_iter", normalized.get("max_iter"))
        _validate_positive_integer("n_init", normalized.get("n_init"))
        if normalized.get("covariance_type") not in {"full", "tied", "diag", "spherical"}:
            raise SearchProtocolError("covariance_type 无效")
        for name, lower in (("reg_covar", 0.0), ("tol", 0.0)):
            value = normalized.get(name)
            if not isinstance(value, (int, float, np.number)) or not np.isfinite(value) or value < lower or (name == "tol" and value == 0):
                raise SearchProtocolError(f"{name} 必须满足有限且大于等于 {lower}")
    elif method == "nn":
        _validate_positive_integer("n_epochs", normalized.get("n_epochs"))
        _validate_positive_integer("n_iter_no_change", normalized.get("n_iter_no_change"))
        for name in ("bandwidth", "lr"):
            value = normalized.get(name)
            if not isinstance(value, (int, float, np.number)) or not np.isfinite(value) or value <= 0:
                raise SearchProtocolError(f"{name} 必须为有限正数")
        alpha = normalized.get("alpha")
        if not isinstance(alpha, (int, float, np.number)) or not np.isfinite(alpha) or alpha < 0:
            raise SearchProtocolError("alpha 必须为有限非负数")
        _validate_hidden_layers(normalized)
    elif method == "nf":
        _validate_positive_integer("transforms", normalized.get("transforms"))
        _validate_positive_integer("bins", normalized.get("bins"))
        _validate_positive_integer("n_epochs", normalized.get("n_epochs"))
        _validate_positive_integer("patience", normalized.get("patience"))
        for name in ("lr",):
            value = normalized.get(name)
            if not isinstance(value, (int, float, np.number)) or not np.isfinite(value) or value <= 0:
                raise SearchProtocolError(f"{name} 必须为有限正数")
        for name in ("weight_decay", "noise_std"):
            value = normalized.get(name)
            if not isinstance(value, (int, float, np.number)) or not np.isfinite(value) or value < 0:
                raise SearchProtocolError(f"{name} 必须为有限非负数")
        if int(normalized["bins"]) < 2:
            raise SearchProtocolError("bins 必须至少为 2")
        _validate_hidden_layers(normalized)
    else:
        for name in ("n_members", "n_epochs", "samples_per_epoch", "ode_steps", "likelihood_batch_size", "patience"):
            _validate_positive_integer(name, normalized.get(name))
        for name in ("lr",):
            value = normalized.get(name)
            if not isinstance(value, (int, float, np.number)) or not np.isfinite(value) or value <= 0:
                raise SearchProtocolError(f"{name} 必须为有限正数")
        for name in ("weight_decay", "noise_std"):
            value = normalized.get(name)
            if not isinstance(value, (int, float, np.number)) or not np.isfinite(value) or value < 0:
                raise SearchProtocolError(f"{name} 必须为有限非负数")
        _validate_hidden_layers(normalized)
    return _json_safe(normalized)


def _validate_hidden_layers(params: dict[str, Any]) -> None:
    layers = params.get("hidden_layer_sizes")
    if layers != [16, 16] and layers != (16, 16):
        raise SearchProtocolError("第一阶段 hidden_layer_sizes 必须固定为 [16, 16]")


def generate_candidates(method: str, search_space: dict[str, Any], fixed: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """按 YAML 中的笛卡尔积生成去重后的候选。"""
    fixed_values = dict(fixed or {})
    keys = list(search_space)
    if not keys:
        raise SearchProtocolError("search_space 不能为空")
    values: list[list[Any]] = []
    for key in keys:
        choices = search_space[key]
        if not isinstance(choices, list) or not choices:
            raise SearchProtocolError(f"search_space.{key} 必须是非空列表")
        values.append(choices)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for combination in itertools.product(*values):
        params = {**fixed_values, **dict(zip(keys, combination))}
        params = validate_method_params(method, params)
        key = _canonical(params)
        if key not in seen:
            candidates.append(params)
            seen.add(key)
    if not candidates:
        raise SearchProtocolError("没有生成候选")
    return candidates


def candidate_id(method: str, params: dict[str, Any]) -> str:
    digest = hashlib.sha256(_canonical({"method": method, "params": params}).encode("utf-8")).hexdigest()
    return f"{method}_{digest[:12]}"


def make_estimator(method: str, params: dict[str, Any], *, kappa: float, random_state: int) -> Any:
    """把一个候选参数完整传给对应 evaluator。"""
    container_kwargs = {"discounted_factor": float(kappa)}
    common = {
        "evaluation_result_container_class": WeightedEvaluationResultContainer,
        "evaluation_result_container_kwargs": container_kwargs,
    }
    if method == "gmm":
        return GMMEvaluator(
            **common,
            n_components=params["n_components"],
            covariance_type=params["covariance_type"],
            reg_covar=params["reg_covar"],
            max_iter=params["max_iter"],
            tol=params["tol"],
            n_init=params["n_init"],
            resample_size=int(params.get("resample_size", 1000)),
            random_state=random_state,
        )
    if method == "nn":
        return GoalSuccessMLPClassifierEvaluator(
            hidden_layer_sizes=tuple(params["hidden_layer_sizes"]),
            hidden_width=int(params["hidden_layer_sizes"][0]),
            bandwidth=params["bandwidth"],
            lr=params["lr"],
            alpha=params["alpha"],
            n_epochs=params["n_epochs"],
            n_iter_no_change=params["n_iter_no_change"],
            early_stopping=bool(params.get("early_stopping", True)),
            validation_fraction=float(params.get("validation_fraction", 0.1)),
            random_state=random_state,
        )
    if method == "nf":
        return NormalizingFlowDensityEvaluator(
            **common,
            transforms=params["transforms"],
            bins=params["bins"],
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            noise_std=params["noise_std"],
            n_epochs=params["n_epochs"],
            patience=params["patience"],
            hidden_layer_sizes=tuple(params["hidden_layer_sizes"]),
            random_state=random_state,
            device="cpu",
        )
    if method == "fm":
        return FMEnsembleEvaluator(
            n_members=params["n_members"],
            kappa=kappa,
            member_kwargs={
                "n_epochs": params["n_epochs"],
                "samples_per_epoch": params["samples_per_epoch"],
                "lr": params["lr"],
                "weight_decay": params["weight_decay"],
                "noise_std": params["noise_std"],
                "patience": params["patience"],
                "ode_steps": params["ode_steps"],
                "likelihood_batch_size": params["likelihood_batch_size"],
                "hidden_layer_sizes": tuple(params["hidden_layer_sizes"]),
                "random_state": random_state,
                "device": "cpu",
            },
        )
    raise SearchProtocolError(f"未知 method: {method}")


def _fit_side(
    method: str, params: dict[str, Any], frame: pd.DataFrame, weights: np.ndarray,
    *, kappa: float, random_state: int, support_goals: np.ndarray | None = None,
) -> FitResult:
    """对一侧数据创建独立实例并拟合，保留 warning 和 evaluator 诊断。"""
    evaluator = make_estimator(method, params, kappa=kappa, random_state=random_state)
    goals = frame.loc[:, list(GOAL_COLUMNS)].to_numpy(dtype=float)
    labels = (frame["termination"].to_numpy() == "reach target").astype(int)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        if method == "nn":
            support = support_goals
            if support is None:
                support = frame.loc[:, list(GOAL_COLUMNS)].drop_duplicates().to_numpy(dtype=float)
            evaluator.fit_classifier(goals, labels, weights, support)
        else:
            evaluator.eval_res_container.add_batch(
                goals, labels.astype(bool).tolist(), [0.0] * len(goals), [0.0] * len(goals),
                [1.0] * len(goals),
            )
            # 容器会为每个 batch 做旧式衰减；搜索协议明确使用绝对时间权重，
            # 因而在插入后覆盖为已计算的权重。
            evaluator.eval_res_container.desired_goal_weights = np.asarray(weights, dtype=float)
            evaluator.fit_evaluator()
    warning_text = [f"{type(item.message).__name__}: {item.message}" for item in captured]
    diagnostics = _json_safe(getattr(evaluator, "fit_diagnostics_", {}))
    return FitResult(evaluator=evaluator, warnings=warning_text, diagnostics=diagnostics)


def _raw_density(evaluator: Any, goals: np.ndarray) -> np.ndarray:
    values = np.asarray(evaluator.evaluate_grid(goals), dtype=float)
    if values.ndim != 1 or len(values) != len(goals):
        raise SearchProtocolError("evaluator 返回的密度形状不正确")
    return values


def calculate_mc_kl(
    reference_evaluator: Any, historical_evaluator: Any, reference: pd.DataFrame,
    mc_indices: list[np.ndarray], *, min_finite_samples: int = 10,
) -> dict[str, Any]:
    """用 reference evaluator 采样计算 KL(reference_full || historical_estimate)。

    每个重复只使用固定的 ``mc_indices`` 首元素作为随机流种子；候选之间共享
    这些随机流。reference 与 historical 仍然是两个独立的同方法实例，且密度
    都在原始目标坐标中计算。该实现不调用 legacy KDE。
    """
    if not mc_indices or any(len(indices) < min_finite_samples for indices in mc_indices):
        return {
            "status": "skipped:insufficient_samples", "kls": [], "finite_ratio": 0.0,
            "finite_count": 0, "total_count": sum(len(indices) for indices in mc_indices),
            "error": "MC 样本数不足",
        }
    kls: list[float] = []
    finite_count = 0
    total_count = 0
    repeat_errors: list[str] = []
    for indices in mc_indices:
        if len(indices) < min_finite_samples:
            repeat_errors.append("MC 样本数不足")
            continue
        points = reference_evaluator.sample(len(indices), int(np.asarray(indices, dtype=np.int64)[0]))
        try:
            q_log = np.asarray(reference_evaluator.evaluate_grid(points, return_log_density=True), dtype=float)
            p_log = np.asarray(historical_evaluator.evaluate_grid(points, return_log_density=True), dtype=float)
            finite = np.isfinite(q_log) & np.isfinite(p_log)
            finite_count += int(np.count_nonzero(finite))
            total_count += len(points)
            if int(np.count_nonzero(finite)) < min_finite_samples:
                repeat_errors.append("有限 MC 点少于下限")
                continue
            value = float(np.mean(q_log[finite] - p_log[finite]))
            if not np.isfinite(value):
                repeat_errors.append("KL 非有限")
            else:
                kls.append(value)
        except Exception as exc:  # evaluator-specific numerical failure is recorded per repeat
            total_count += len(points)
            repeat_errors.append(f"{type(exc).__name__}: {exc}")
    if not kls:
        status = "invalid:nonfinite_kl" if total_count else "skipped:insufficient_samples"
        return {
            "status": status, "kls": [], "finite_ratio": finite_count / max(total_count, 1),
            "finite_count": finite_count, "total_count": total_count,
            "error": "; ".join(repeat_errors)[:2000],
        }
    return {
        "status": "ok", "kls": kls, "finite_ratio": finite_count / max(total_count, 1),
        "finite_count": finite_count, "total_count": total_count,
        "error": "; ".join(repeat_errors)[:2000],
    }


def _candidate_record(
    candidate: dict[str, Any], candidate_name: str, method: str, stage: str,
    data: TrajectoryData, *, kappa: float, global_seed: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    history_goals, history_labels, history_weights = _history_arrays(
        data.history, data.checkpoint, kappa
    )
    history_frame = pd.DataFrame(history_goals, columns=list(GOAL_COLUMNS))
    history_frame["termination"] = np.where(history_labels == 1, "reach target", "timeout")
    reference = data.reference
    record: dict[str, Any] = {
        "candidate_id": candidate_name,
        "method": method,
        "stage": stage,
        "task": data.task,
        "seed": data.seed,
        "checkpoint": data.checkpoint,
        "phase": data.phase,
        "status": "error",
        "mean_kl": float("nan"),
        "median_kl": float("nan"),
        "finite_result_ratio": 0.0,
        "skipped_count": 0,
        "history_rows": int(len(history_frame)),
        "history_successes": int(np.sum(history_labels == 1)),
        "reference_rows": int(len(reference)),
        "reference_successes": int((reference["termination"] == "reach target").sum()),
        "fit_warnings": [],
        "runtime_s": 0.0,
        "error": "",
        "sample_hash": data.sample_hash,
        "mc_index_hash": _hash_indices(data.mc_indices),
    }
    try:
        base_seed = _stable_seed(global_seed, method, candidate_name, data.task, data.seed, data.checkpoint)
        support = reference.loc[:, list(GOAL_COLUMNS)].drop_duplicates().to_numpy(dtype=float)
        history_fit = _fit_side(
            method, candidate, history_frame, history_weights,
            kappa=kappa, random_state=_stable_seed(base_seed, "history"), support_goals=support,
        )
        # reference 必须是同方法、同参数的第二个独立实例。
        reference_fit = _fit_side(
            method, candidate, reference, np.ones(len(reference), dtype=float),
            kappa=kappa, random_state=_stable_seed(base_seed, "reference"), support_goals=support,
        )
        record["fit_warnings"] = (
            history_fit.warnings + reference_fit.warnings
            + [f"历史侧：{value}" for value in history_fit.diagnostics.get("quality_warnings", [])]
            + [f"参考侧：{value}" for value in reference_fit.diagnostics.get("quality_warnings", [])]
        )
        mc = calculate_mc_kl(reference_fit.evaluator, history_fit.evaluator, reference, data.mc_indices)
        record["status"] = mc["status"]
        record["mean_kl"] = float(np.mean(mc["kls"])) if mc["kls"] else float("nan")
        record["median_kl"] = float(np.median(mc["kls"])) if mc["kls"] else float("nan")
        record["finite_result_ratio"] = float(mc["finite_ratio"])
        record["error"] = mc.get("error", "")
        record["history_fit_diagnostics"] = history_fit.diagnostics
        record["reference_fit_diagnostics"] = reference_fit.diagnostics
    except Exception as exc:
        record["status"] = _classify_failure(exc)
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["skipped_count"] = 1
    record["runtime_s"] = round(time.perf_counter() - started, 4)
    return _json_safe(record)


def _classify_failure(exc: Exception) -> str:
    text = str(exc).lower()
    if "sample" in text or "success" in text or "class" in text or "empty" in text:
        return "skipped:insufficient_samples"
    if "finite" in text or "nan" in text or "inf" in text:
        return "invalid:nonfinite_fit"
    return "error:fit_failed"


def run_candidate(
    method: str, stage: str, params: dict[str, Any], datasets: list[TrajectoryData],
    *, kappa: float, global_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """运行一个候选并同时返回汇总行和逐轨迹详情。"""
    name = candidate_id(method, params)
    records = [
        _candidate_record(params, name, method, stage, data, kappa=kappa, global_seed=global_seed)
        for data in datasets
    ]
    def finite_value(value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return float("nan")
        return number if np.isfinite(number) else float("nan")

    valid = [value for record in records if np.isfinite(value := finite_value(record["mean_kl"]))]
    phases = {
        phase: [value for record in records if record["phase"] == phase
                and np.isfinite(value := finite_value(record["mean_kl"]))]
        for phase in ("early", "middle", "late")
    }
    by_seed: dict[int, list[float]] = {}
    for record in records:
        value = finite_value(record["mean_kl"])
        if np.isfinite(value):
            by_seed.setdefault(int(record["seed"]), []).append(value)
    trajectory_means = [float(np.mean(values)) for values in by_seed.values() if values]
    status_values = [record["status"] for record in records]
    if valid and all(value == "ok" for value in status_values):
        status = "ok"
    elif valid:
        status = "partial"
    elif any(value.startswith("error") for value in status_values):
        status = "error"
    else:
        status = "skipped"
    summary = {
        "candidate_id": candidate_id(method, params),
        "method": method,
        "stage": stage,
        "status": status,
        "mean_kl": float(np.mean(valid)) if valid else float("nan"),
        "median_kl": float(np.median(valid)) if valid else float("nan"),
        "worst_trajectory_kl": max(trajectory_means) if trajectory_means else float("nan"),
        "early_kl": float(np.mean(phases["early"])) if phases["early"] else float("nan"),
        "middle_kl": float(np.mean(phases["middle"])) if phases["middle"] else float("nan"),
        "late_kl": float(np.mean(phases["late"])) if phases["late"] else float("nan"),
        "finite_result_ratio": float(np.mean([
            finite_value(record["finite_result_ratio"]) for record in records
        ])) if records else 0.0,
        "skipped_count": int(sum(value != "ok" for value in status_values)),
        "fit_warnings": sorted({warning for record in records for warning in record.get("fit_warnings", [])}),
        "runtime_s": round(sum(float(record["runtime_s"]) for record in records), 4),
        "params_json": _canonical(params),
        "error": "; ".join(record["error"] for record in records if record.get("error"))[:4000],
    }
    details = {
        "candidate_id": candidate_id(method, params),
        "method": method,
        "stage": stage,
        "params": _json_safe(params),
        "records": records,
        "protocol": {
            "kl": "KL(reference_full || historical_estimate)",
            "mc_points": "uniform rows from current fixed-eval grid with reference density importance normalization",
            "legacy_kde_search_target": False,
        },
    }
    return _json_safe(summary), _json_safe(details)


def _phase(position: int, total: int) -> str:
    if total <= 1:
        return "middle"
    fraction = position / (total - 1)
    return "early" if fraction < 1 / 3 else "late" if fraction >= 2 / 3 else "middle"


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise SearchProtocolError(f"配置根节点必须为 mapping: {path}")
    return value


def _default_config(task: str, method: str) -> Path:
    return ROOT / "configs" / "evaluate" / "offline" / task / "hyperparameter_search" / f"{method}.yaml"


def _select_candidates(
    method: str, config: dict[str, Any], stage: str, candidate_file: Path | None,
    max_candidates: int | None, top_k: int | None,
) -> list[dict[str, Any]]:
    if stage == "validation" and candidate_file:
        frame = pd.read_csv(candidate_file)
        if "params_json" not in frame.columns:
            raise SearchProtocolError(f"candidate_results 缺少 params_json: {candidate_file}")
        frame = frame[frame.get("status", "ok").isin(["ok", "partial"])].copy()
        frame = frame[np.isfinite(frame["mean_kl"].to_numpy(dtype=float))]
        frame = frame.sort_values(["mean_kl", "candidate_id"], kind="stable")
        selected = []
        for value in frame["params_json"].head(top_k or 5):
            selected.append(validate_method_params(method, json.loads(value)))
        if not selected:
            raise SearchProtocolError("candidate_results 没有可供正式复核的有限候选")
        return selected
    stage_config = config.get("stages", {}).get(stage, {})
    candidates = generate_candidates(method, config.get("search_space", {}), config.get("fixed", {}))
    limit = max_candidates or stage_config.get("max_candidates")
    if limit is not None:
        _validate_positive_integer("max_candidates", int(limit))
        candidates = candidates[: int(limit)]
    return candidates


def _write_yaml(path: Path, value: Any) -> None:
    path.write_text(yaml.safe_dump(_json_safe(value), allow_unicode=True, sort_keys=False), encoding="utf-8")


def _append_csv(path: Path, row: dict[str, Any], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([{column: row.get(column) for column in columns}], columns=columns)
    if path.exists() and path.stat().st_size:
        existing = pd.read_csv(path)
        if "candidate_id" in columns and "candidate_id" in existing:
            existing = existing[existing["candidate_id"].astype(str) != str(row.get("candidate_id"))]
        elif {"candidate_id", "task", "seed", "checkpoint"}.issubset(columns) and not existing.empty:
            mask = np.ones(len(existing), dtype=bool)
            for key in ("candidate_id", "task", "seed", "checkpoint"):
                mask &= existing[key].astype(str).to_numpy() == str(row.get(key))
            existing = existing.loc[~mask]
        pd.concat([existing, frame], ignore_index=True).to_csv(path, index=False)
    else:
        frame.to_csv(path, index=False)


def _read_existing_candidate_ids(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    frame = pd.read_csv(path)
    if "candidate_id" not in frame:
        return set()
    status = frame.get("status", pd.Series("ok", index=frame.index)).astype(str)
    return set(frame.loc[status.isin(["ok", "partial", "skipped"]), "candidate_id"].astype(str))


def _upsert_details(path: Path, details: dict[str, Any]) -> None:
    """按 candidate_id 更新 JSONL，重复候选不会积累不可审计的重复详情。"""
    rows: list[dict[str, Any]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    rows = [row for row in rows if row.get("candidate_id") != details.get("candidate_id")]
    rows.append(details)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_input_hashes(path: Path, paths: Iterable[Path], data_root: Path) -> None:
    mapping = {}
    for source in sorted(set(paths)):
        try:
            key = str(source.relative_to(data_root))
        except ValueError:
            key = str(source)
        mapping[key] = _sha256_file(source)
    path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _git_metadata() -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        result["revision"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True
        ).stdout.strip()
        result["status"] = subprocess.run(
            ["git", "status", "--short"], cwd=ROOT, text=True, capture_output=True, check=True
        ).stdout.splitlines()
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _source_hashes() -> dict[str, str]:
    paths = [
        Path(__file__),
        ROOT / "src/gc_ope/evaluate/evaluator_base.py",
        ROOT / "src/gc_ope/evaluate/evaluator_gmm.py",
        ROOT / "src/gc_ope/evaluate/evaluator_nn.py",
        ROOT / "src/gc_ope/evaluate/evaluator_nf.py",
        ROOT / "src/gc_ope/evaluate/evaluator_fm.py",
    ]
    return {str(path.relative_to(ROOT)): _sha256_file(path) for path in paths}


def run_search(
    *, method: str, task: str, stage: str, config_path: Path, output_dir: Path,
    data_root: Path = ROOT, seeds: list[int] | None = None, checkpoints: list[int] | None = None,
    candidate_file: Path | None = None, max_candidates: int | None = None,
    top_k: int | None = None, samples_per_checkpoint: int | None = None,
    mc_samples: int | None = None, mc_repeats: int | None = None,
    sampling_seed: int | None = None, mc_seed: int | None = None,
    global_seed: int | None = None, resume: bool = True,
) -> dict[str, Any]:
    if method not in METHODS or task not in TASKS or stage not in {"screening", "validation"}:
        raise SearchProtocolError("method/task/stage 参数无效")
    config = _load_config(config_path)
    if config.get("method") != method:
        raise SearchProtocolError(f"配置 method 与入口不一致: {config.get('method')} != {method}")
    protocol = dict(config.get("protocol", {}))
    stage_config = dict(config.get("stages", {}).get(stage, {}))
    actual_seeds = [int(value) for value in (seeds or stage_config.get("seeds") or protocol.get("seeds", [1, 2, 3, 4, 5]))]
    actual_sample_count = int(samples_per_checkpoint or protocol.get("samples_per_checkpoint", 100))
    actual_mc_samples = int(mc_samples or stage_config.get("mc_samples") or protocol.get("mc_samples", 10000))
    actual_mc_repeats = int(mc_repeats or stage_config.get("mc_repeats") or protocol.get("mc_repeats", 5))
    actual_sampling_seed = int(sampling_seed if sampling_seed is not None else protocol.get("sampling_seed", 0))
    actual_mc_seed = int(mc_seed if mc_seed is not None else protocol.get("mc_seed", 0))
    actual_global_seed = int(global_seed if global_seed is not None else protocol.get("global_seed", 0))
    kappa = float(protocol.get("kappa", 0.9))
    if not 0 < kappa <= 1:
        raise SearchProtocolError("kappa 必须在 (0,1] 内")
    if actual_sample_count <= 0 or actual_mc_samples <= 0 or actual_mc_repeats <= 0:
        raise SearchProtocolError("抽样和 MC 配置必须为正数")

    datasets: list[TrajectoryData] = []
    all_paths: list[Path] = []
    for seed in actual_seeds:
        available = [step for step, _ in checkpoint_files(data_root, task, seed)]
        selected = _choose_checkpoints(available, stage_config, checkpoints)
        for position, checkpoint in enumerate(selected):
            data = load_trajectory(
                data_root, task, seed, checkpoint,
                samples_per_checkpoint=actual_sample_count,
                sampling_seed=actual_sampling_seed,
                mc_samples=actual_mc_samples,
                mc_repeats=actual_mc_repeats,
                mc_seed=actual_mc_seed,
                phase=_phase(position, len(selected)),
            )
            datasets.append(data)
            all_paths.extend(data.input_paths)
    if not datasets:
        raise SearchProtocolError("没有加载任何轨迹")

    candidates = _select_candidates(method, config, stage, candidate_file, max_candidates, top_k)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "candidate_results.csv"
    details_path = output_dir / "candidate_details.jsonl"
    summary_path = output_dir / "validation_summary.csv"
    log_path = output_dir / "run.log"
    run_config = {
        "method": method, "task": task, "stage": stage,
        "config_path": str(config_path), "config": config,
        "protocol_effective": {
            "seeds": actual_seeds, "samples_per_checkpoint": actual_sample_count,
            "mc_samples": actual_mc_samples, "mc_repeats": actual_mc_repeats,
            "sampling_seed": actual_sampling_seed, "mc_seed": actual_mc_seed,
            "global_seed": actual_global_seed, "kappa": kappa,
            "include_current_checkpoint": True,
            "fixed_eval_only": True,
            "reference_fit": "same_method_same_candidate_parameters_on_current_full_csv",
            "kl_direction": "reference_full || historical_estimate",
            "legacy_kde_direct_search_target": False,
        },
        "candidate_count": len(candidates),
        "candidates": candidates,
        "source_hashes": _source_hashes(),
    }
    snapshot_path = output_dir / "search_config.yaml"
    if snapshot_path.exists():
        previous = _load_config(snapshot_path)
        if _canonical(previous) != _canonical(run_config):
            raise SearchProtocolError(
                f"输出目录已有不同的 search_config，拒绝混合续跑: {output_dir}"
            )
    _write_yaml(output_dir / "search_config.yaml", run_config)
    input_hash_path = output_dir / "input_hashes.json"
    if input_hash_path.exists():
        previous_hashes = json.loads(input_hash_path.read_text(encoding="utf-8"))
        current_hashes = {}
        for source in sorted(set(all_paths)):
            try:
                key = str(source.relative_to(data_root))
            except ValueError:
                key = str(source)
            current_hashes[key] = _sha256_file(source)
        if previous_hashes != current_hashes:
            raise SearchProtocolError("输入 CSV 哈希已变化，拒绝混合续跑")
    _write_input_hashes(input_hash_path, all_paths, Path(data_root))
    metadata = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "git": _git_metadata(),
        "config_sha256": hashlib.sha256(_canonical(run_config).encode("utf-8")).hexdigest(),
        "input_file_count": len(set(all_paths)),
        "dataset_count": len(datasets),
        "source_hashes": _source_hashes(),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(_json_safe(metadata), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    existing = _read_existing_candidate_ids(candidate_path) if resume else set()
    with log_path.open("a", encoding="utf-8") as log:
        def log_line(message: str) -> None:
            line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()

        log_line(f"开始 {method}/{task}/{stage}，候选 {len(candidates)}，轨迹 {len(datasets)}")
        for params in candidates:
            name = candidate_id(method, params)
            if name in existing:
                log_line(f"断点续跑跳过已完成候选 {name}")
                continue
            log_line(f"运行候选 {name}: {_canonical(params)}")
            summary, details = run_candidate(
                method, stage, params, datasets, kappa=kappa, global_seed=actual_global_seed
            )
            _append_csv(candidate_path, summary, RESULT_COLUMNS)
            _upsert_details(details_path, details)
            for record in details["records"]:
                _append_csv(summary_path, record, SUMMARY_COLUMNS)
            existing.add(name)
            log_line(f"候选完成 {name}: status={summary['status']} mean_kl={summary['mean_kl']}")

    if candidate_path.exists():
        results = pd.read_csv(candidate_path)
        valid = results[np.isfinite(results["mean_kl"].to_numpy(dtype=float))]
        if not valid.empty:
            best = valid.sort_values(["mean_kl", "candidate_id"], kind="stable").iloc[0]
            best_params = json.loads(best["params_json"])
            _write_yaml(output_dir / "best_params.yaml", {
                "method": method, "stage": stage, "candidate_id": best["candidate_id"],
                "status": "ready_for_manual_confirmation", "mean_kl": float(best["mean_kl"]),
                "params": best_params,
            })
        else:
            _write_yaml(output_dir / "best_params.yaml", {
                "method": method, "stage": stage, "status": "no_finite_candidate", "params": None,
            })
    return {
        "output_dir": str(output_dir), "candidate_count": len(candidates),
        "dataset_count": len(datasets), "results_path": str(candidate_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="离线 GMM/NN/NF/FM 超参数搜索")
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--stage", choices=("screening", "validation"), default="screening")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--candidate-file", type=Path)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--checkpoints", type=int, nargs="+")
    parser.add_argument("--samples-per-checkpoint", type=int)
    parser.add_argument("--mc-samples", type=int)
    parser.add_argument("--mc-repeats", type=int)
    parser.add_argument("--sampling-seed", type=int)
    parser.add_argument("--mc-seed", type=int)
    parser.add_argument("--global-seed", type=int)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config or _default_config(args.task, args.method)
    output_dir = args.output_dir or ROOT / "logs" / "hyperparameter_search" / args.task / args.method / args.stage
    result = run_search(
        method=args.method, task=args.task, stage=args.stage, config_path=config_path,
        output_dir=output_dir, seeds=args.seeds, checkpoints=args.checkpoints,
        candidate_file=args.candidate_file, max_candidates=args.max_candidates, top_k=args.top_k,
        samples_per_checkpoint=args.samples_per_checkpoint, mc_samples=args.mc_samples,
        mc_repeats=args.mc_repeats, sampling_seed=args.sampling_seed, mc_seed=args.mc_seed,
        global_seed=args.global_seed, resume=not args.no_resume,
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
