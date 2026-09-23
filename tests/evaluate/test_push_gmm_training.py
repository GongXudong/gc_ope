"""对齐旧正式训练配置，并验证真实评估回调能够驱动 GMM 课程。"""

import json
import os
import shlex
import subprocess
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[2]
SHELL_DIR = ROOT / "scripts/train_policy/shells/my_push/curriculum"
LAUNCHER = SHELL_DIR / "sac_omega_gmm.sh"


def training_overrides(path, prefix):
    """直接读取可执行命令，防止文档种子与启动脚本互相脱节。"""
    commands = []
    for line in path.read_text().splitlines():
        if line.startswith(prefix):
            tokens = shlex.split(line)
            commands.append(tokens[tokens.index("scripts/train_policy/train.py") + 1:])
    return commands


def training_config(overrides):
    with initialize_config_dir(config_dir=str(ROOT / "configs/train"), version_base=None):
        return compose(config_name="config", overrides=overrides)


@pytest.mark.parametrize("seed_index", range(5))
def test_five_configs_only_replace_estimator_and_output_name(seed_index):
    old = training_overrides(SHELL_DIR / "sac_omega.sh", "uv run")[:5]
    new = training_overrides(LAUNCHER, "run_seed ")
    assert len(new) == 5
    # 先统一输出名再解析路径；其余旧参数必须逐项相等。
    old_cfg = training_config(old[seed_index])
    new_cfg = training_config(new[seed_index])
    new_cfg.experiment_name = old_cfg.experiment_name
    old_data = OmegaConf.to_container(old_cfg, resolve=True)
    new_data = OmegaConf.to_container(new_cfg, resolve=True)
    estimator = new_data["env"]["curriculum_kwargs"].pop("estimator_config")
    assert new_data == old_data
    offline = json.loads((ROOT / "configs/evaluate/push_same_family_all100.json").read_text())
    assert estimator == {"method": "gmm", "parameters": offline["parameters"]["gmm"]}
    assert new_cfg.callback[0].evaluate_nums_in_callback * new_cfg.env.callback_env.num_process == 96


def test_launcher_dry_run_and_seed_selection(tmp_path):
    # 从仓库以外启动，确认脚本自己定位工作目录；dry-run 不启动训练。
    all_seeds = subprocess.run(["bash", str(LAUNCHER), "--dry-run"], cwd=tmp_path,
                               capture_output=True, text=True, check=True)
    assert all_seeds.stdout.count("即将运行 seed") == 5
    one_seed = subprocess.run(["bash", str(LAUNCHER), "--seed", "3", "--dry-run"],
                              capture_output=True, text=True, check=True)
    assert one_seed.stdout.count("即将运行 seed") == 1
    assert "algo.seed=26" in one_seed.stdout
    invalid = subprocess.run(["bash", str(LAUNCHER), "--seed", "0"], capture_output=True)
    assert invalid.returncode == 2


def test_launcher_saves_legacy_process_log_including_child_output(tmp_path):
    # 在临时仓库复用真实启动函数，把百万步训练换成仅打印的子进程。
    launcher = tmp_path / LAUNCHER.relative_to(ROOT)
    launcher.parent.mkdir(parents=True)
    body = LAUNCHER.read_text().split("\nrun_seed 1 ", 1)[0]
    child = "print('find min: point [0.1 0.2 0.02] with score 0.5')"
    probe = ("import os, subprocess, sys; "
             "assert os.environ['PYTHONUNBUFFERED'] == '1'; "
             f"subprocess.run([sys.executable, '-c', {child!r}], check=True); "
             "print('stderr retained', file=sys.stderr)")
    launcher.write_text(body + "\nrun_seed 1 conda run --no-capture-output -n gc_ope python -c "
                        + shlex.quote(probe) + "\n")
    subprocess.run(["bash", str(launcher)], capture_output=True, text=True, check=True,
                   env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    name = "omega_gmm_dscnt_0_9_b_0_n_100_eval_96_seed_1"
    process_log = tmp_path / f"logs_in_process/my_push/sac/my_push_sac_{name}.txt"
    console_log = tmp_path / f"logs/my_push/sac/{name}/console.log"
    text = process_log.read_text()
    assert text == console_log.read_text()
    assert "find min: point [0.1 0.2 0.02] with score 0.5" in text
    assert "stderr retained" in text
    # 重复启动不能截断之前的完整课程日志。
    retry = subprocess.run(["bash", str(launcher)], capture_output=True)
    assert retry.returncode != 0
    assert process_log.read_text() == text


def test_real_callback_updates_weights_fits_gmm_and_trains_sac(tmp_path, monkeypatch, capsys):
    import gymnasium as gym
    import torch
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from gc_ope.env.get_vec_env import get_vec_env  # 注册真实 Push 环境。
    from gc_ope.algorithm.get_algorithm import get_algo
    from gc_ope.algorithm.get_callbacks import get_callback_list
    from gc_ope.algorithm.curriculum.omega_wrapper import OMEGAWrapper

    torch.set_num_threads(1)
    cfg = training_config(training_overrides(LAUNCHER, "run_seed ")[0])
    wrapper = OMEGAWrapper(gym.make("MyPushSparse-v0"),
                          **OmegaConf.to_container(cfg.env.curriculum_kwargs, resolve=True))
    train_env = DummyVecEnv([lambda: Monitor(wrapper)])
    # 仅此测试把评估任务设为两步、近距离目标，以确定性地覆盖“已成功后拟合”分支。
    # 成功标签仍来自真实物理环境，未注入伪造的 evaluation records。
    easy_env = gym.make("MyPushSparse-v0", max_episode_steps=2)
    goals = np.array([[x, y, .02] for x in [-.02, -.01, .01, .02]
                      for y in [-.02, -.01, .01, .02]])
    goal_index = iter(range(10000))
    monkeypatch.setattr(easy_env.unwrapped.task, "_sample_goal",
                        lambda: goals[next(goal_index) % len(goals)].copy())
    eval_env = DummyVecEnv([lambda: Monitor(easy_env)])
    fit_rounds = []
    original_fit = wrapper.estimator.fit_evaluator

    def record_fit():
        fit_rounds.append(wrapper.estimator.eval_res_container.desired_goal_weights.copy())
        return original_fit()

    monkeypatch.setattr(wrapper.estimator, "fit_evaluator", record_fit)
    try:
        assert not wrapper.estimator.eval_res_container.desired_goal_list
        # 测试缩短优化与评估间隔，但两轮回调仍各收集 96 条真实 episode 结果。
        cfg.algo.device = "cpu"
        cfg.algo.learning_starts = 4
        cfg.algo.batch_size = 4
        cfg.algo.buffer_size = 64
        cfg.algo.net_arch = [16, 16]
        cfg.callback[0].eval_freq = 8
        cfg.callback[0].best_model_save_path = str(tmp_path)
        cfg.callback[0].log_path = str(tmp_path)
        cfg.callback[1].save_path = str(tmp_path)
        cfg.callback[1].save_checkpoint_every_n_timesteps = 16
        callbacks = get_callback_list(cfg.callback, cfg.env, eval_env, train_env)
        assert callbacks[0].n_eval_episodes == 96
        model = get_algo(cfg.algo, train_env)
        model.learn(total_timesteps=16, callback=callbacks)
        assert model._n_updates > 0
        assert len(fit_rounds) == 2
        np.testing.assert_allclose(fit_rounds[0], np.ones(96))
        np.testing.assert_allclose(fit_rounds[1], np.r_[np.full(96, .9), np.ones(96)])
        assert wrapper.estimator_ready
        assert wrapper.estimator.model.fit_diagnostics_["fitting_scheme"] == "direct_weighted_em"
        assert wrapper.estimator.model.gmm.reg_covar == .05
        assert np.isfinite(wrapper.kl_value)
        assert 0 <= wrapper.alpha <= 1
        data = np.load(tmp_path / "evaluations.npz")
        np.testing.assert_array_equal(data["timesteps"], [8, 16])
        assert data["results"].shape == (2, 96)
        assert list(tmp_path.glob("*_16_steps.zip"))
        assert list(tmp_path.glob("*replay_buffer_16_steps.pkl"))

        # 强制选择 OMEGA 的 MEGA 分支，确认返回的是环境候选目标，而非 GMM 直接生成点。
        from gc_ope.env.utils import desired_goal_utils
        candidates = np.array([[x, y, .02] for x in np.linspace(-.2, .2, 10)
                               for y in np.linspace(-.2, .2, 10)])
        candidate_iter = iter(candidates.copy())
        monkeypatch.setattr(desired_goal_utils, "sample_a_desired_goal", lambda env: next(candidate_iter))
        monkeypatch.setattr(np.random, "rand", lambda: 1.)
        scores = wrapper.estimator.evaluate(candidates, return_density=True)[1]
        valid = scores >= wrapper.p_ag_density_threshold
        expected = np.argmin(np.where(valid, scores, np.inf)) if valid.any() else np.argmax(scores)
        np.testing.assert_array_equal(wrapper.sample_goal(), candidates[expected])
        # 真正 GMM 选点产生的旧格式文本可由师兄的解析器直接读取。
        from gc_ope.utils.train_log_process import process_file
        process_log = tmp_path / "process.txt"
        process_log.write_text(capsys.readouterr().out)
        timestamps, goals, scores = process_file(
            process_log, sample_goal_log_begin_strs=["find min", "find max"])
        assert len(goals) >= 1
        np.testing.assert_allclose(goals[-1], candidates[expected], atol=1e-8)
        assert np.isfinite(scores[-1])
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
