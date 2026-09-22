"""与保存实验版本做数值对照；未提供基线路径时显式跳过。"""

import importlib.util
import os
from pathlib import Path
import numpy as np
import pytest

from gc_ope.evaluate.evaluator_factory import make_evaluator
from gc_ope.evaluate.offline_data import EvaluationBatch


@pytest.mark.parametrize("method,class_name", [
    ("gmm", "GMMEvaluator"), ("nn", "GoalSuccessMLPClassifierEvaluator"),
    ("nf", "NormalizingFlowDensityEvaluator"), ("fm", "FlowMatchingDensityEvaluator"),
])
def test_preserved_fitting_algorithm(method, class_name):
    baseline = os.environ.get("GC_OPE_BASELINE")
    if not baseline:
        pytest.skip("需显式提供 GC_OPE_BASELINE，防止意外导入旧目录")
    source = Path(baseline) / "src/gc_ope/evaluate" / f"evaluator_{method}.py"
    spec = importlib.util.spec_from_file_location(f"baseline_{method}", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, class_name)
    from gc_ope.evaluate.evaluator_factory import DEFAULT_PARAMETERS
    from gc_ope.evaluate.evaluation_result_container import WeightedEvaluationResultContainer
    parameters = dict(DEFAULT_PARAMETERS[method])
    rng = np.random.default_rng(5)
    goals = rng.normal(size=(100, 2))
    labels = goals[:, 0] > 0
    weights = np.linspace(.3, 1, 100)
    axis = np.linspace(-3, 3, 12)
    support = np.array(np.meshgrid(axis, axis)).reshape(2, -1).T
    if method == "nn":
        # 新早停是有意修正；这里只核对关闭早停后的原分类器算法。
        parameters = {"hidden_width": 16, "n_epochs": 100, "early_stopping": False,
                      "random_state": 0, "bandwidth": .2}
    current = make_evaluator(method, support_goals=support, parameters=parameters)
    EvaluationBatch(goals, labels, weights).fill(current)
    current.fit_evaluator()
    if method == "nn":
        # NN 关闭早停后的分类器保持不变；连续密度定义是已授权的改动，不要求伪等价。
        parameters.pop("bandwidth")
        old = cls(**parameters).fit_classifier(goals, labels, weights, support)
        np.testing.assert_allclose(current.predict_success_probability(support),
                                   old.predict_success_probability(support), rtol=1e-8)
    else:
        old = cls(evaluation_result_container_class=WeightedEvaluationResultContainer,
                  evaluation_result_container_kwargs={"discounted_factor": .9}, **parameters)
        EvaluationBatch(goals, labels, weights).fill(old)
        old.fit_evaluator()
        np.testing.assert_allclose(current.evaluate(goals, return_density=False)[1],
                                   old.evaluate(goals, return_density=False)[1], atol=1e-6)
