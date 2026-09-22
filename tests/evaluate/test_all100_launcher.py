"""全量调度的小规模实跑；只用临时合成数据，不启动正式实验。"""

import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import psutil


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run_push_all100.py"


def arguments(tmp_path, *, slow=False):
    # 每个 seed 只有一个 checkpoint，每方法只训练两次，MC 只采 16 个点。
    for seed in range(1, 6):
        directory = tmp_path / f"checkpoints/my_push/sac/seed_{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        goals = np.random.default_rng(seed).normal(size=(100, 2))
        pd.DataFrame(dict(x=goals[:, 0], y=goals[:, 1], z=.02,
                          termination=np.where(np.arange(100) % 2, "reach target", "timeout"))).to_csv(
            directory / "rl_model_10000_steps_eval_res_on_fixed.csv", index=False)
    config = {"mc_samples": 16, "mc_repeats": 1, "parameters": {
        "nn": {"n_epochs": 100000 if slow else 2, "early_stopping": False},
        "fm": {"n_epochs": 2, "samples_per_epoch": 16, "hidden_features": 4, "ode_steps": 2},
        "nf": {"n_epochs": 2, "hidden_features": 4, "transforms": 1},
        "gmm": {"n_components": 2},
    }}
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(config))
    return [sys.executable, str(SCRIPT), "--checkpoint-root", str(tmp_path / "checkpoints"),
            "--output", str(tmp_path / "result"), "--config", str(path), "--checkpoints", "10000"]


def test_real_launcher_5_seeds_4_workers_sequence_resume_and_config(tmp_path):
    args = arguments(tmp_path)
    dry = subprocess.run([*args, "--dry-run"], capture_output=True, text=True, timeout=20)
    assert dry.returncode == 0, dry.stderr
    assert "计算 worker 上限：20" in dry.stdout
    assert dry.stdout.count("--workers 4") == 20
    assert not (tmp_path / "result").exists()
    first = subprocess.run(args, capture_output=True, text=True, timeout=120)
    assert first.returncode == 0, first.stdout + first.stderr
    output = tmp_path / "result"
    audit = json.loads((output / "audit.json").read_text())
    assert audit["status"] == "complete"
    assert len(audit["audits"]) == 20
    assert all(value["ok"] == 1 and not value["invalid"] for value in audit["audits"].values())
    # 每个方法必须先启动全部五 seed；全部完成后下一个方法才能启动。
    lines = first.stdout.splitlines()
    for current, following in zip(["nn", "fm", "nf"], ["fm", "nf", "gmm"]):
        starts = [index for index, line in enumerate(lines) if f"启动 {current} seed=" in line]
        ends = [index for index, line in enumerate(lines) if f"完成 {current} seed=" in line]
        next_start = next(index for index, line in enumerate(lines) if f"启动 {following} seed=" in line)
        assert len(starts) == len(ends) == 5
        assert max(starts) < min(ends) and max(ends) < next_start
    original = {path.name: path.read_bytes() for path in (output / "method_per_seed").glob("*.csv")}
    assert len(original) == 20
    second = subprocess.run(args, capture_output=True, text=True, timeout=120)
    assert second.returncode == 0, second.stdout + second.stderr
    assert original == {path.name: path.read_bytes() for path in (output / "method_per_seed").glob("*.csv")}
    # 总入口也检查配置，不能用旧结果跳过新参数。
    settings = tmp_path / "settings.json"
    data = json.loads(settings.read_text())
    data["mc_samples"] = 17
    settings.write_text(json.dumps(data))
    changed = subprocess.run(args, capture_output=True, text=True, timeout=20)
    assert changed.returncode != 0 and "续跑配置" in changed.stderr


def test_missing_input_fails_before_creating_output(tmp_path):
    args = arguments(tmp_path)
    missing = tmp_path / "checkpoints/my_push/sac/seed_5/rl_model_10000_steps_eval_res_on_fixed.csv"
    missing.unlink()
    result = subprocess.run(args, capture_output=True, text=True, timeout=20)
    assert result.returncode != 0
    assert not (tmp_path / "result").exists()


def test_nn_only_preserves_5_by_4_parallelism(tmp_path):
    args = arguments(tmp_path)
    result = subprocess.run([*args, "--methods", "nn", "--dry-run"], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("--methods nn") == 5
    assert result.stdout.count("--workers 4") == 5
    assert "--methods fm" not in result.stdout


def test_weighted_em_runs_as_separate_method(tmp_path):
    args = arguments(tmp_path)
    settings = tmp_path / "settings.json"
    config = json.loads(settings.read_text())
    config["protocol"] = "push_same_family_gmm_em_v1"
    config["parameters"]["gmm_em"] = {"n_components": 2}
    settings.write_text(json.dumps(config))
    result = subprocess.run([*args, "--methods", "gmm_em"], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    outputs = list((tmp_path / "result/method_per_seed").glob("*.csv"))
    assert len(outputs) == 5 and all(p.name.startswith("gmm_em_") for p in outputs)
    assert json.loads((tmp_path / "result/audit.json").read_text())["status"] == "complete"


def test_signal_cleans_seed_processes_and_workers(tmp_path):
    args = arguments(tmp_path, slow=True)
    with (tmp_path / "launch.log").open("w") as log:
        process = subprocess.Popen(args, stdout=log, stderr=log)
        children = []
        try:
            deadline = time.monotonic() + 40
            while time.monotonic() < deadline:
                children = psutil.Process(process.pid).children(recursive=True)
                active_logs = list((tmp_path / "result/runs/nn").glob("seed_*/jobs/*.log"))
                if len(active_logs) == 5 and len(children) >= 25:
                    break
                assert process.poll() is None
                time.sleep(.1)
            assert len(active_logs) == 5 and len(children) >= 25
            # 第二个总入口不能启动同一批任务。
            duplicate = subprocess.run(args, capture_output=True, text=True, timeout=20)
            assert duplicate.returncode != 0 and "已有运行占用" in duplicate.stderr
            process.send_signal(signal.SIGINT)
            assert process.wait(timeout=25) == 130
            _, alive = psutil.wait_procs(children, timeout=5)
            assert not [child for child in alive if child.status() != psutil.STATUS_ZOMBIE]
            assert json.loads((tmp_path / "result/audit.json").read_text())["status"] == "interrupted"
            assert not (tmp_path / "result/runs/fm").exists()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            for child in children:
                if child.is_running():
                    child.kill()


def test_failed_seed_stops_stage_and_cleans_other_seeds(tmp_path, monkeypatch):
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location("all100_launcher", SCRIPT)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    # 模拟一个 seed 异常退出，其余四个持续运行；必须停止整个阶段并回收。
    def failing_command(args, method, seed, config_path):
        code = "raise SystemExit(7)" if seed == 1 else "import time; time.sleep(120)"
        return [sys.executable, "-c", code]
    monkeypatch.setattr(launcher, "seed_command", failing_command)
    args = SimpleNamespace(output=tmp_path, seeds=[1, 2, 3, 4, 5], workers_per_seed=4, checkpoints=[10000])
    import pytest
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="退出码 7"):
        launcher.run_stage(args, "nn", tmp_path / "config.json", lambda message: None)
    assert time.monotonic() - started < 20
