"""正式命名不能改变已选定算法；显式隐藏层应真实决定网络结构。"""

import json
from pathlib import Path
import numpy as np
import pytest
from gc_ope.evaluate.evaluator_factory import DEFAULT_PARAMETERS, make_evaluator
from gc_ope.evaluate.offline_data import EvaluationBatch
from gc_ope.evaluate.offline_experiment import ExperimentConfig


@pytest.mark.parametrize("method", ["nn", "gmm", "nf", "fm"])
def test_final_methods_match_saved_selected_versions(method):
    # 合成数据和预期值由重命名前的 8af1bb7 版本产生，不依赖本地大实验目录。
    fixture = json.loads((Path(__file__).parent / "fixtures/final_methods_parity.json").read_text())
    data = {key: np.asarray(value) for key, value in fixture["inputs"].items()}
    model = make_evaluator(method, support_goals=data["x"])
    EvaluationBatch(data["x"], data["y"], data["w"]).fill(model)
    model.fit_evaluator()
    expected = fixture["expected"][method]
    np.testing.assert_allclose(model.log_density(data["query"]), expected["density"], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(model.sample(100, 23), expected["samples"], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("method", ["gmm_em", "nf_reg", "fm_reg", "fm_ensemble"])
def test_retired_method_names_fail_explicitly(method):
    with pytest.raises(ValueError, match="未知"):
        make_evaluator(method)
    with pytest.raises(ValueError, match="退役"):
        ExperimentConfig("unused", parameters={method: {}})


def test_only_five_methods_and_formal_config_agree():
    config = json.loads((Path(__file__).resolve().parents[2] /
                        "configs/evaluate/push_same_family_all100.json").read_text())
    assert set(DEFAULT_PARAMETERS) == {"kde", "gmm", "nn", "nf", "fm"}
    assert config["parameters"] == DEFAULT_PARAMETERS
    assert config["parameters"]["nf"]["hidden_layer_sizes"] == [16, 16]
    assert config["parameters"]["fm"]["member_parameters"]["hidden_layer_sizes"] == [32, 32, 32]


@pytest.mark.parametrize("layers", [16, [], [0, 8], [8, True], [8, 3.5]])
@pytest.mark.parametrize("method", ["nf", "fm"])
def test_invalid_hidden_layers_fail_at_construction(method, layers):
    parameters = ({"hidden_layer_sizes": layers} if method == "nf" else
                  {"member_parameters": {"hidden_layer_sizes": layers}})
    with pytest.raises(ValueError, match="隐藏层|hidden_layer_sizes"):
        make_evaluator(method, parameters=parameters)


def test_nonuniform_layers_reach_nf_and_fm_networks():
    import torch
    nf = make_evaluator("nf", parameters={"hidden_layer_sizes": [7, 5, 3]})
    model = nf._new_model()
    # NSF 的自回归网络由 Zuko 构造；所有层宽确实出现在内部网络，而非只保存在配置。
    widths = [layer.out_features for layer in model.modules() if isinstance(layer, torch.nn.Linear)]
    assert widths[:3] == [7, 5, 3]
    from gc_ope.evaluate.evaluator_fm import _FMMember
    member = _FMMember(hidden_layer_sizes=[7, 5])
    widths = [layer.out_features for layer in member._new_model() if isinstance(layer, torch.nn.Linear)]
    assert widths == [7, 5, 2]
