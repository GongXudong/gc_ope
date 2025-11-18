from typing import Callable
from functools import partial
from omegaconf import DictConfig

from stable_baselines3.common.evaluation import evaluate_policy
from gc_ope.algorithm.utils.my_evaluate_policy import evaluate_policy_with_success_rate


def get_evaluate_method(eval_cfg: DictConfig, env_cfg: DictConfig) -> Callable:
    if env_cfg.env_id.startswith("FlyCraft"):
        return get_evaluate_method_for_flycraft(eval_cfg)
    elif env_cfg.env_id.startswith("MyReach"):
        return get_evaluate_method_for_my_reach(eval_cfg)
    elif env_cfg.env_id.startswith("MyPointMaze") or env_cfg.env_id.startswith("MyAntMaze"):
        return get_evaluate_method_for_point_maze(eval_cfg)
    else:
        raise ValueError(f"Can not get evaluation method for env: {env_cfg.env_id}!")

def get_evaluate_method_for_flycraft(eval_cfg: DictConfig) -> Callable:
    if eval_cfg.eval_mode == "success_rate":
        return evaluate_policy_with_success_rate
    elif eval_cfg.eval_mode == "cumulative_return":
        return evaluate_policy
    else:
        raise ValueError(f"Can not process evaluation model: {eval_cfg.eval_mode}!")

def get_evaluate_method_for_my_reach(eval_cfg: DictConfig) -> Callable:
    if eval_cfg.eval_mode == "success_rate":
        return evaluate_policy_with_success_rate
    elif eval_cfg.eval_mode == "cumulative_return":
        return evaluate_policy
    else:
        raise ValueError(f"Can not process evaluation model: {eval_cfg.eval_mode}!")

def get_evaluate_method_for_point_maze(eval_cfg: DictConfig) -> Callable:
    if eval_cfg.eval_mode == "success_rate":
        return partial(evaluate_policy_with_success_rate, success_key_in_info="success")
    elif eval_cfg.eval_mode == "cumulative_return":
        return evaluate_policy
    else:
        raise ValueError(f"Can not process evaluation model: {eval_cfg.eval_mode}!")
