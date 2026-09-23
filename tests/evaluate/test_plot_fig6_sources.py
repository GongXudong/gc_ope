"""正式五方法的来源映射、原始数据保真及论文绘图样式。"""

import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sources(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[2] / "scripts/plot_fig6.py"
    spec = importlib.util.spec_from_file_location("fig6", script)
    plot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plot)
    monkeypatch.setattr(plot, "EXPECTED", [10000, 20000])
    sources = []
    for name, method in zip(plot.METHODS, ["kde", "gmm_em", "nn", "fm_ensemble", "nf_reg"]):
        root = tmp_path / name
        (root / "method_per_seed").mkdir(parents=True)
        source = dict(name=name, method=method, root=str(root),
                      protocol="legacy" if name=="KDE" else "test_protocol", sha256={})
        for seed in range(1, 6):
            if name == "KDE":
                pd.DataFrame({"evaluation_index": plot.EXPECTED, "$D_{KL}$ [his]": [.01, .02]}).to_csv(
                    root / f"my_push_sac_seed_{seed}_kde_0_9_eval_res_in_training.csv", index=False)
            else:
                pd.DataFrame([dict(task="push", method=method, seed=seed, checkpoint=step,
                    protocol="test_protocol", kl_mode="raw", status="ok", error="",
                    kl=seed*step/1000000, kl_seed_std=.001, fit_quality="warning", fit_warnings="测试提醒")
                    for step in plot.EXPECTED]).to_csv(root / "method_per_seed" / f"{method}_push_seed{seed}.csv", index=False)
        sources.append(source)
    return plot, sources


def test_only_final_names_and_raw_values_preserved(sources):
    plot, sources = sources
    data, audit, hashes = plot.load_data(sources)
    assert len(data) == 50 and set(data.method) == set(plot.METHODS)
    assert set(data.loc[data.method=="GMM", "source_method"]) == {"gmm_em"}
    assert len(audit["NN_seed1"]["quality_warnings"]) == 2
    assert len(hashes) == 25
    summary = plot.summarize(data)
    assert summary.loc[(summary.method=="NN") & (summary.checkpoint==10000), "mean"].iloc[0] == pytest.approx(.03)


def test_reject_missing_or_wrong_identity(sources):
    plot, sources = sources
    path = Path(sources[1]["root"]) / "method_per_seed/gmm_em_push_seed1.csv"
    original = pd.read_csv(path)
    original.iloc[:1].to_csv(path,index=False)
    with pytest.raises(ValueError,match="覆盖率"):
        plot.load_data(sources)
    original["method"] = "gmm"
    original.to_csv(path,index=False)
    with pytest.raises(ValueError,match="协议不一致"):
        plot.load_data(sources)


def test_hash_and_exact_five_sources_required(sources):
    plot, sources = sources
    with pytest.raises(ValueError,match="五个来源"):
        plot.load_data(sources[:-1])
    sources[1]["sha256"] = {"method_per_seed/gmm_em_push_seed1.csv": "invalid"}
    with pytest.raises(ValueError,match="摘要"):
        plot.load_data(sources)


def test_paper_labels_solid_lines_and_default_palette(sources, tmp_path, monkeypatch):
    plot, sources = sources
    data, _, _ = plot.load_data(sources)
    from matplotlib.figure import Figure
    checked = []
    def inspect(fig, *args, **kwargs):
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Environment Steps"
        assert ax.get_ylabel() == r"$D_{KL}(P_{ag} \| \tilde{p}_{ag})$"
        assert ax.get_legend().get_title().get_text() == "Estimation Method"
        assert [line.get_label() for line in ax.lines] == plot.METHODS
        assert all(line.get_linestyle()=="-" for line in ax.lines)
        np.testing.assert_allclose([line.get_color() for line in ax.lines], plot.sns.color_palette("deep",5))
        checked.append(True)
    monkeypatch.setattr(Figure,"savefig",inspect)
    plot.draw(data,plot.summarize(data),tmp_path,.5)
    assert len(checked) == 4
