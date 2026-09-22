"""新 NN 单独重跑后的绘图来源检查；合成数据只写到 pytest 临时目录。"""

import importlib.util
import json
from pathlib import Path
import pandas as pd
import pytest


@pytest.fixture
def sources(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[2] / "scripts/plot_fig6.py"
    spec = importlib.util.spec_from_file_location("fig6", script)
    plot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plot)
    monkeypatch.setattr(plot, "EXPECTED", [10000, 20000])
    old, new, legacy = [tmp_path / name for name in ["old", "new", "legacy"]]
    for root in [old, new]:
        (root / "method_per_seed").mkdir(parents=True)
        for method in (["nn"] if root == new else ["nn", "gmm", "nf", "fm"]):
            for seed in range(1, 6):
                pd.DataFrame([dict(task="push", method=method, seed=seed, checkpoint=step,
                    protocol="push_same_family_nn_logloss_v2" if root == new else "push_same_family_inclusive_v1",
                    kl_mode="raw", status="ok", error="", kl=.02 if root == new else .5,
                    kl_seed_std=.001, fit_quality="warning", fit_warnings="测试提醒")
                    for step in plot.EXPECTED]).to_csv(root / "method_per_seed" / f"{method}_push_seed{seed}.csv", index=False)
    legacy.mkdir()
    for seed in range(1, 6):
        pd.DataFrame({"evaluation_index": plot.EXPECTED, "$D_{KL}$ [his]": [.01, .02]}).to_csv(
            legacy / f"my_push_sac_seed_{seed}_kde_0_9_eval_res_in_training.csv", index=False)
    return plot, old, new, legacy


def test_override_uses_new_nn_and_keeps_quality_warnings(sources):
    plot, old, new, legacy = sources
    frame, audit, hashes = plot.load_data(old, legacy, new)
    assert len(frame) == 50
    assert (frame.loc[frame.method == "NN", "kl"] == .02).all()
    assert (frame.loc[frame.method == "GMM", "kl"] == .5).all()
    assert all(str(new) in path for path in frame.loc[frame.method == "NN", "source"])
    assert len(audit["nn_seed1"]["quality_warnings"]) == 2
    assert len(hashes) == 25


def test_override_rejects_old_nn_protocol(sources):
    plot, old, new, legacy = sources
    with pytest.raises(ValueError, match="协议不一致"):
        plot.load_data(old, legacy, old)


def test_missing_nn_checkpoint_prevents_plot(sources):
    plot, old, new, legacy = sources
    path = new / "method_per_seed/nn_push_seed5.csv"
    pd.read_csv(path).iloc[:1].to_csv(path, index=False)
    with pytest.raises(ValueError, match="覆盖率"):
        plot.load_data(old, legacy, new)


def test_weighted_em_adds_sixth_method_without_replacing_gmm(sources, tmp_path):
    plot, old, new, legacy = sources
    em = tmp_path / "em"
    (em / "method_per_seed").mkdir(parents=True)
    for seed in range(1, 6):
        frame = pd.read_csv(old / "method_per_seed" / f"gmm_push_seed{seed}.csv")
        frame["method"], frame["protocol"], frame["kl"] = "gmm_em", "push_same_family_gmm_em_v1", .03
        frame.to_csv(em / "method_per_seed" / f"gmm_em_push_seed{seed}.csv", index=False)
    data, audit, _ = plot.load_data(old, legacy, new, em)
    assert len(data) == 60 and data.method.nunique() == 6
    assert (data.loc[data.method == "GMM", "kl"] == .5).all()
    assert (data.loc[data.method == "GMM (weighted EM)", "kl"] == .03).all()


def test_regularized_em_keeps_original_and_checks_configuration(sources, tmp_path):
    plot, old, new, legacy = sources
    roots = [tmp_path / "em", tmp_path / "reg"]
    for root, reg, kl in zip(roots, [1e-6, .05], [.03, .01]):
        (root / "method_per_seed").mkdir(parents=True)
        (root / "experiment.json").write_text(json.dumps({"mc_samples": 10000,
            "parameters": {"gmm_em": {"reg_covar": reg, "n_components": 5}}}))
        for seed in range(1, 6):
            frame = pd.read_csv(old / "method_per_seed" / f"gmm_push_seed{seed}.csv")
            frame["method"], frame["protocol"], frame["kl"] = "gmm_em", "push_same_family_gmm_em_v1", kl
            frame.to_csv(root / "method_per_seed" / f"gmm_em_push_seed{seed}.csv", index=False)
    data, audit, hashes = plot.load_data(old, legacy, new, *roots)
    assert data.method.nunique() == 7 and len(data) == 70
    assert (data.loc[data.method == "GMM (weighted EM)", "kl"] == .03).all()
    assert (data.loc[data.method == plot.REGULARIZED_EM, "kl"] == .01).all()
    assert audit["gmm_em_seed1"]["ok"] == audit["gmm_em_reg005_seed1"]["ok"] == 2
    assert len(hashes) == 37  # 35 份 CSV，加上两份变体配置。
    with pytest.raises(ValueError, match="图例不符"):
        plot.load_data(old, legacy, new, roots[0], roots[0])
    path = roots[1] / "experiment.json"
    config = json.loads(path.read_text())
    config["mc_samples"] = 10
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="还有差异"):
        plot.load_data(old, legacy, new, *roots)
