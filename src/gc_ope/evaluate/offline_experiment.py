"""一个 checkpoint 的离线比较：读数据、独立拟合两个模型、计算连续 KL。"""

from dataclasses import dataclass, field
import time
import traceback
import numpy as np

from gc_ope.evaluate.offline_data import load_pair
from gc_ope.evaluate.evaluator_factory import make_evaluator, fit_evaluator, DEFAULT_PARAMETERS
from gc_ope.evaluate.evaluator_common import InsufficientSamples
from gc_ope.evaluate.utils.distribution_kl import monte_carlo_kl


@dataclass
class ExperimentConfig:
    checkpoint_root: str
    kappa: float = 0.9
    samples_per_checkpoint: int = 100
    sampling_seed: int = 0
    mc_samples: int = 10000
    mc_repeats: int = 5
    mc_seed: int = 0
    kl_mode: str = "raw"
    parameters: dict = field(default_factory=lambda: {key: dict(value) for key, value in DEFAULT_PARAMETERS.items()})
    protocol: str = "push_same_family_nn_logloss_v2"

    def __post_init__(self):
        if not 0 < self.kappa <= 1 or not np.isfinite(self.kappa):
            raise ValueError("时间折扣必须在 (0,1] 内")
        if min(self.samples_per_checkpoint, self.mc_samples, self.mc_repeats) <= 0:
            raise ValueError("样本数量和 MC 重复次数必须为正")
        if self.kl_mode not in {"raw", "legacy"}:
            raise ValueError("未知 KL 模式")


def fit_pair(method, history, reference, config):
    """两侧同类、同配置、独立实例；NN 的几何网格不包含额外标签。"""
    support = np.unique(reference.goals, axis=0)
    estimate_model = make_evaluator(method, kappa=config.kappa, support_goals=support,
                                    parameters=config.parameters.get(method))
    reference_model = make_evaluator(method, kappa=config.kappa, support_goals=support,
                                     parameters=config.parameters.get(method))
    history.fill(estimate_model)
    reference.fill(reference_model)
    # 样本不足时标明发生在哪一侧，不能偷偷换成 KDE 参考分布。
    for side, model in [("历史", estimate_model), ("参考", reference_model)]:
        try:
            fit_evaluator(method, model)
        except InsufficientSamples as exc:
            raise InsufficientSamples(f"{side}侧：{exc}") from exc
    return estimate_model, reference_model


def run_checkpoint(config, method, seed, checkpoint):
    start = time.perf_counter()
    row = dict(task="push", seed=seed, checkpoint=checkpoint, method=method,
               kl=None, kl_seed_std=None, historical_successes=0, reference_successes=0,
               historical_records=0, reference_records=0, status="error", error="",
               protocol=config.protocol, kl_mode=config.kl_mode)
    row.update(fit_quality="not_checked", fit_warnings="")
    detail = {}
    try:
        history, reference = load_pair(
            config.checkpoint_root, seed, checkpoint, kappa=config.kappa,
            sampling_seed=config.sampling_seed, n_samples=config.samples_per_checkpoint,
        )
        row.update(historical_successes=int(history.successes.sum()),
                   reference_successes=int(reference.successes.sum()),
                   historical_records=len(history.goals), reference_records=len(reference.goals))
        estimate_model, reference_model = fit_pair(method, history, reference, config)
        # 计算成功与拟合质量分开记录：不删除质量差的点来美化曲线。
        warnings = [f"{side}侧：{message}" for side, model in
                    [("历史", estimate_model), ("参考", reference_model)]
                    for message in getattr(model, "fit_diagnostics_", {}).get("quality_warnings", [])]
        if method in {"gmm_em", "nf_reg", "fm_reg", "fm_ensemble"} or (method == "nn" and estimate_model.early_stopping):
            row["fit_quality"] = "warning" if warnings else "passed_checks"
        row["fit_warnings"] = "；".join(warnings)
        metric = monte_carlo_kl(reference_model, estimate_model, config.mc_samples,
                                config.mc_repeats, config.mc_seed, config.kl_mode)
        row.update(kl=metric["kl"], kl_seed_std=metric["kl_seed_std"], status="ok")
        detail = {**metric, "estimate_fit": getattr(estimate_model, "fit_diagnostics_", {}),
                  "reference_fit": getattr(reference_model, "fit_diagnostics_", {})}
    except InsufficientSamples as exc:
        row.update(status="skipped:insufficient_samples", error=str(exc))
    except Exception:
        # 保留完整堆栈；调度器仍会写入本条结果，最终以非零退出码提示失败。
        row["error"] = traceback.format_exc()
    row["job_time_s"] = time.perf_counter() - start
    return row, detail
