"""把二维能力估计器接到旧 Push wrapper 的 XYZ 目标接口。"""

import numpy as np
from gc_ope.evaluate.evaluation_result_container import WeightedEvaluationResultContainer
from gc_ope.evaluate.evaluator_factory import make_evaluator, fit_evaluator
from gc_ope.evaluate.evaluator_common import uniform_grid_kl


class PlanarEvaluator:
    """环境仍保存和采样 XYZ 目标，仅密度计算投影到 XY 平面。"""

    def __init__(self, method, support_goals, kappa=0.9, parameters=None):
        self.method = method
        self.eval_res_container = WeightedEvaluationResultContainer(kappa)
        support = np.asarray(support_goals, dtype=float)
        if support.ndim != 2 or support.shape[1] != 3 or np.ptp(support[:, 2]) > 1e-6:
            raise ValueError("平面适配器只支持固定高度的 XYZ 目标")
        self.height = support[0, 2]
        self.model = make_evaluator(method, kappa=kappa, support_goals=support[:, :2], parameters=parameters)

    def fit_evaluator(self):
        source = self.eval_res_container
        goals = np.asarray(source.desired_goal_list, dtype=float)
        if goals.ndim != 2 or goals.shape[1] != 3 or not np.allclose(goals[:, 2], self.height, atol=1e-6):
            raise ValueError("收到非平面目标，不能静默丢弃 z 坐标")
        # 不修改环境容器；模型持有独立的 XY 副本及原始时间权重。
        target = self.model.eval_res_container
        target.reset()
        target.add_batch(goals[:, :2].tolist(), source.success_list,
                         source.cumulative_reward_list, source.discounted_cumulative_reward_list,
                         source.desired_goal_weights.tolist())
        _, scaled, weights, density = fit_evaluator(self.method, self.model)
        return goals[np.asarray(source.success_list, dtype=bool)], scaled, weights, density

    def evaluate(self, desired_goals, scale=True, return_density=True):
        goals = np.asarray(desired_goals, dtype=float)
        if scale:
            goals = goals[:, :2]
        return self.model.evaluate(goals, scale, return_density)

    def kl_divergence_uniform_to_kde_integrate(self, samples, dV, u_density):
        # 仍使用旧 OMEGA 的均匀分布网格积分；离线两模型 KL 才使用 MC。
        return uniform_grid_kl(self.model.log_density(np.asarray(samples)[:, :2]), dV, u_density)
