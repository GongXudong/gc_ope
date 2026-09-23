"""离线比较与在线课程学习共用的估计器构造入口。"""

import numpy as np
from gc_ope.evaluate.evaluation_result_container import WeightedEvaluationResultContainer
from gc_ope.evaluate.evaluator_common import InsufficientSamples


# 正式五方法的已选定参数；与离线配置对应，在线构造也采用同一版本。
DEFAULT_PARAMETERS = {'nn': {'n_epochs': 500,
        'lr': 0.001,
        'alpha': 0.0001,
        'random_state': 0,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'n_iter_no_change': 30,
        'bandwidth': 0.2,
        'hidden_layer_sizes': [16, 16],
        'min_epochs': 50,
        'tol': 0.0001},
 'fm': {'n_members': 3,
        'random_state': 0,
        'member_parameters': {'n_epochs': 500,
                              'samples_per_epoch': 2000,
                              'lr': 0.001,
                              'weight_decay': 0.0,
                              'ode_steps': 32,
                              'likelihood_batch_size': 1024,
                              'device': 'cpu',
                              'noise_std': 0.1,
                              'hidden_layer_sizes': [32, 32, 32]}},
 'nf': {'n_epochs': 300,
        'transforms': 2,
        'bins': 8,
        'lr': 0.001,
        'weight_decay': 0.0,
        'noise_std': 0.2,
        'early_stopping': True,
        'validation_fraction': 0.15,
        'min_epochs': 30,
        'patience': 50,
        'validation_interval': 10,
        'tol': 0.0001,
        'fallback_epochs': 50,
        'random_state': 0,
        'device': 'cpu',
        'hidden_layer_sizes': [16, 16]},
 'gmm': {'n_components': 5,
         'covariance_type': 'full',
         'n_init': 1,
         'max_iter': 200,
         'tol': 0.001,
         'reg_covar': 0.05,
         'random_state': 0},
 'kde': {'kde_bandwidth': 0.2}}

def make_evaluator(method, *, kappa=0.9, support_goals=None, parameters=None):
    """每次调用构造一个全新模型；两侧配置相同但不共享已拟合参数。"""
    from gc_ope.evaluate.evaluator_kde import KDEEvaluator
    from gc_ope.evaluate.evaluator_gmm import GMMEvaluator
    from gc_ope.evaluate.evaluator_nn import GoalSuccessMLPClassifierEvaluator
    from gc_ope.evaluate.evaluator_nf import NormalizingFlowDensityEvaluator
    from gc_ope.evaluate.evaluator_fm import FlowMatchingDensityEvaluator
    classes = dict(kde=KDEEvaluator, gmm=GMMEvaluator, nn=GoalSuccessMLPClassifierEvaluator,
                   nf=NormalizingFlowDensityEvaluator, fm=FlowMatchingDensityEvaluator)
    if method not in classes:
        raise ValueError(f"未知估计方法：{method}")
    from copy import deepcopy
    kwargs = {**deepcopy(DEFAULT_PARAMETERS[method]), **(parameters or {})}
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
    if method in {"nf", "fm"} and count < 2:
        raise InsufficientSamples(f"{method} 至少需要两个成功样本")
    return estimator.fit_evaluator()
