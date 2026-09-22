"""离线比较与在线课程学习共用的估计器构造入口。"""

import numpy as np
from gc_ope.evaluate.evaluation_result_container import WeightedEvaluationResultContainer
from gc_ope.evaluate.evaluator_common import InsufficientSamples


# 保留保存版本的参数；FM 采用已完成内部调参的 500 次更新。
DEFAULT_PARAMETERS = {
    "kde": {"kde_bandwidth": 0.2},
    "gmm": {"n_components": 5, "resample_size": 1000, "random_state": 0},
    "gmm_em": {"n_components": 5, "covariance_type": "full", "random_state": 0},
    "nn": {"hidden_layer_sizes": [16, 16], "n_epochs": 500, "bandwidth": 0.2, "random_state": 0},
    "nf": {"n_epochs": 100, "hidden_features": 32, "random_state": 0},
    "fm": {"n_epochs": 500, "hidden_features": 32, "samples_per_epoch": 2000, "random_state": 0},
    "nf_reg": {"random_state": 0},
    "fm_reg": {"random_state": 0},
    "fm_ensemble": {"n_members": 3, "random_state": 0},
}


def make_evaluator(method, *, kappa=0.9, support_goals=None, parameters=None):
    """每次调用构造一个全新模型；两侧配置相同但不共享已拟合参数。"""
    from gc_ope.evaluate.evaluator_kde import KDEEvaluator
    from gc_ope.evaluate.evaluator_gmm import GMMEvaluator
    from gc_ope.evaluate.evaluator_gmm_em import WeightedGMMEvaluator
    from gc_ope.evaluate.evaluator_nn import GoalSuccessMLPClassifierEvaluator
    from gc_ope.evaluate.evaluator_nf import NormalizingFlowDensityEvaluator
    from gc_ope.evaluate.evaluator_fm import FlowMatchingDensityEvaluator
    from gc_ope.evaluate.evaluator_regularized_flow import RegularizedNFEvaluator, RegularizedFMEvaluator
    from gc_ope.evaluate.evaluator_flow_ensemble import FlowEnsembleEvaluator
    classes = dict(kde=KDEEvaluator, gmm=GMMEvaluator, gmm_em=WeightedGMMEvaluator, nn=GoalSuccessMLPClassifierEvaluator,
                   nf=NormalizingFlowDensityEvaluator, fm=FlowMatchingDensityEvaluator,
                   nf_reg=RegularizedNFEvaluator, fm_reg=RegularizedFMEvaluator, fm_ensemble=FlowEnsembleEvaluator)
    if method not in classes:
        raise ValueError(f"未知估计方法：{method}")
    kwargs = {**DEFAULT_PARAMETERS[method], **(parameters or {})}
    if method == "nn":
        if "hidden_width" in (parameters or {}) and "hidden_layer_sizes" not in parameters:
            kwargs.pop("hidden_layer_sizes")
        kwargs["support_goals"] = support_goals
    return classes[method](
        evaluation_result_container_class=WeightedEvaluationResultContainer,
        evaluation_result_container_kwargs={"discounted_factor": kappa}, **kwargs,
    )


def fit_evaluator(method, estimator):
    """明确区分样本不足与实现/数值错误；后者必须报错，不能当作跳过。"""
    flags = np.asarray(estimator.eval_res_container.success_list, dtype=bool)
    count = int(flags.sum())
    if count == 0:
        raise InsufficientSamples("没有成功样本")
    if method in {"nf", "fm", "nf_reg", "fm_reg", "fm_ensemble"} and count < 2:
        raise InsufficientSamples(f"{method} 至少需要两个成功样本")
    return estimator.fit_evaluator()
