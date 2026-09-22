"""从真实 CLI 验证续跑、配置隔离以及 SIGTERM 后工作进程清理。"""

from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import psutil


ROOT = Path(__file__).resolve().parents[2]


def command(tmp_path, method="gmm", extra_config=None):
    inputs = tmp_path / "checkpoints/my_push/sac/seed_1"
    inputs.mkdir(parents=True, exist_ok=True)
    goals = np.random.default_rng(3).normal(size=(100, 2))
    pd.DataFrame(dict(x=goals[:, 0], y=goals[:, 1],
                      termination=np.where(np.arange(100) % 2, "reach target", "timeout"))).to_csv(
        inputs / "rl_model_10000_steps_eval_res_on_fixed.csv", index=False)
    config = tmp_path / "settings.json"
    config.write_text(json.dumps({"mc_samples": 32, "mc_repeats": 1, **(extra_config or {})}))
    return [sys.executable, str(ROOT / "scripts/evaluate_push_estimators.py"),
            "--checkpoint-root", str(tmp_path / "checkpoints"), "--output", str(tmp_path / "result"),
            "--methods", method, "--seeds", "1", "--checkpoints", "10000", "--workers", "1",
            "--config", str(config)]


def test_cli_resume_and_changed_config_rejection(tmp_path):
    args = command(tmp_path)
    first = subprocess.run(args, capture_output=True, text=True, timeout=40)
    assert first.returncode == 0, first.stdout + first.stderr
    csv = tmp_path / "result/gmm_push_seed1.csv"
    original = csv.read_bytes()
    second = subprocess.run(args, capture_output=True, text=True, timeout=40)
    assert second.returncode == 0, second.stderr
    assert "待运行 0" in second.stdout
    assert csv.read_bytes() == original
    config = tmp_path / "settings.json"
    values = json.loads(config.read_text())
    values["mc_samples"] = 33
    config.write_text(json.dumps(values))
    changed = subprocess.run(args, capture_output=True, text=True, timeout=40)
    assert changed.returncode != 0 and "续跑配置" in changed.stderr
    assert csv.read_bytes() == original


def test_sigterm_cleans_children_and_releases_lock(tmp_path):
    from gc_ope.evaluate.offline_results import result_lock
    args = command(tmp_path, "fm", {"parameters": {"fm": {"n_epochs": 100000}}})
    with (tmp_path / "process.log").open("w") as log:
        process = subprocess.Popen(args, stdout=log, stderr=log)
        children = []
        try:
            deadline = time.monotonic() + 30
            job_log = tmp_path / "result/jobs/fm_seed1_10000.log"
            while time.monotonic() < deadline:
                if job_log.exists() and job_log.stat().st_size:
                    children = psutil.Process(process.pid).children(recursive=True)
                    break
                assert process.poll() is None, (tmp_path / "process.log").read_text()
                time.sleep(.05)
            assert children, "未看到工作进程启动"
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=20) == 130
            _, alive = psutil.wait_procs(children, timeout=5)
            assert not [child for child in alive if child.status() != psutil.STATUS_ZOMBIE]
            with result_lock(tmp_path / "result"):
                pass
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            for child in children:
                if child.is_running():
                    child.kill()
