"""真实 Push 环境和短 SAC 优化检查；数据注入只验证接口，不代表学习效果。"""

import numpy as np
import pytest
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from gc_ope.env.get_vec_env import get_vec_env  # 沿用项目的环境注册过程。
from gc_ope.algorithm.curriculum.omega_wrapper import OMEGAWrapper
from gc_ope.evaluate.evaluator_kde import KDEEvaluator


PARAMETERS = {
    "gmm": {"n_components": 2},
    "gmm_em": {"n_components": 2},
    "nn": {"n_epochs": 2, "early_stopping": False},
    "nf": {"n_epochs": 1, "hidden_features": 8, "transforms": 2},
    "fm": {"n_epochs": 1, "hidden_features": 8, "samples_per_epoch": 32, "ode_steps": 2},
}


def make_env(method):
    return OMEGAWrapper(
        gym.make("MyPushSparse-v0"), sample_n=4,
        eval_kl_u_p_args={"step_list": [.1, .1, .02]},
        estimator_config={"method": method, "parameters": PARAMETERS[method]},
    )


def evaluation_records():
    rng = np.random.default_rng(1)
    return [dict(desired_goal=[*xy, .02], success=bool(index % 2), cumulative_reward=-1.)
            for index, xy in enumerate(rng.uniform(-.2, .2, (24, 2)))]


@pytest.mark.parametrize("method", PARAMETERS)
def test_real_push_and_sac_can_use_estimator(method, tmp_path):
    import torch
    torch.set_num_threads(1)
    wrapper = make_env(method)
    vec = DummyVecEnv([lambda: wrapper])
    try:
        # 无成功样本及不足两个正样本时，都能退回合法的环境目标采样。
        observation = vec.reset()
        assert observation["desired_goal"].shape == (1, 3)
        wrapper.sync_evaluation_stat(evaluation_records())
        info = wrapper.get_info_to_log()
        assert wrapper.estimator_ready
        assert np.isfinite(info["KL[p_dg|p_ag]"])
        assert 0 <= info["alpha"] <= 1
        model = SAC("MultiInputPolicy", vec, learning_starts=4, batch_size=4,
                    buffer_size=64, policy_kwargs={"net_arch": [16, 16]}, device="cpu", seed=0)
        model.learn(total_timesteps=16)
        assert model._n_updates > 0
        model.save(tmp_path / f"sac_{method}")
        # 清空容器后不能继续使用过期能力分布。
        wrapper.reset_evaluation_result_container()
        wrapper.get_info_to_log()
        assert not wrapper.estimator_ready
        vec.reset()
    finally:
        vec.close()


@pytest.mark.parametrize("method", ["nn", "nf", "fm"])
def test_early_sample_shortage_falls_back(method):
    wrapper = make_env(method)
    try:
        wrapper.sync_evaluation_stat([dict(desired_goal=[.1, .1, .02], success=True, cumulative_reward=0.)])
        wrapper.reset()
        assert not wrapper.estimator_ready
        assert wrapper.get_info_to_log()["alpha"] == 0
    finally:
        wrapper.close()


def test_default_kde_is_unchanged():
    wrapper = OMEGAWrapper(gym.make("MyPushSparse-v0"), sample_n=4)
    try:
        assert type(wrapper.estimator) is KDEEvaluator
        assert wrapper.estimator.kde.bandwidth == .2
    finally:
        wrapper.close()


def test_subprocess_env_receives_statistics_and_returns_info():
    from functools import partial
    vec = SubprocVecEnv([partial(make_env, "gmm")], start_method="spawn")
    try:
        vec.reset()
        vec.env_method("sync_evaluation_stat", evaluation_records())
        assert np.isfinite(vec.env_method("get_info_to_log")[0]["KL[p_dg|p_ag]"])
        vec.step(np.zeros((1, *vec.action_space.shape)))
    finally:
        vec.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
