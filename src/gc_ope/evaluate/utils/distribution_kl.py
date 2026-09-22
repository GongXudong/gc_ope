"""同一目标坐标下，两个连续分布之间的蒙特卡洛 KL。"""

import numpy as np


def monte_carlo_kl(reference, estimate, n_samples=10000, repeats=5, random_state=0,
                   mode="raw"):
    """计算 KL(当前全量参考 || 历史估计)，单位为 nat。

    每次从参考模型重新采样，取 log p - log q 的均值。负的有限 MC 估计值
    不截断。legacy 模式只用于核对旧算法：各自标准化空间直接比较并过滤
    q<1e-10 的样本；它与原始目标空间 KL 不同，不能混在同一结果中。
    """
    if n_samples <= 0 or repeats <= 0 or mode not in {"raw", "legacy"}:
        raise ValueError("MC 数量、重复次数或模式无效")
    values, kept = [], []
    for repeat in range(repeats):
        seed = int(np.random.SeedSequence([random_state, repeat]).generate_state(1)[0])
        goals = reference.sample(n_samples, seed)
        if mode == "raw":
            log_p = reference.log_density(goals)
            log_q = estimate.log_density(goals)
            mask = np.ones(n_samples, dtype=bool)
        else:
            scaled = reference.scaler.transform(goals)
            _, log_p = reference.evaluate(scaled, scale=False, return_density=False)
            _, log_q = estimate.evaluate(scaled, scale=False, return_density=False)
            mask = log_q > np.log(1e-10)
        if not np.isfinite(log_p).all() or not np.isfinite(log_q).all() or not mask.any():
            raise FloatingPointError("MC 对数密度非有限或过滤后没有样本")
        values.append(float(np.mean(log_p[mask] - log_q[mask])))
        kept.append(int(mask.sum()))
    return {"kl": float(np.mean(values)), "kl_seed_std": float(np.std(values)),
            "kl_repeats": values, "mc_kept": kept}
