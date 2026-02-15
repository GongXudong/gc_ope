"""
Evaluate π_e on π_b's goal
统一接口，根据环境类型自动路由到对应的评估函数
"""

from pathlib import Path
from typing import Optional, Union
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common import type_aliases

from gc_ope.algorithm.ope.logged_dataset import LoggedDataset
from gc_ope.algorithm.ope.evaluation.evaluate_my_reach import evaluate_agent as evaluate_my_reach
from gc_ope.algorithm.ope.evaluation.evaluate_my_push_slide import evaluate_agent as evaluate_my_push_slide
from gc_ope.algorithm.ope.evaluation.evaluate_flycraft import evaluate_agent as evaluate_flycraft

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent.parent

#TODO：可以用类的形式来实现？更优雅一些？
def evaluate_agent(
    eval_pi_ckpt_path: Union[str, Path, "type_aliases.PolicyPredictor"],
    data_behavior: LoggedDataset,
    behavior_pi_name: str,
    algo_type: str,
    env_cfg: Optional[DictConfig] = None,
    process_num: int = 4,
    gamma: float = 0.995,
    seed: int = 42,
    logger=None
) -> tuple[float, float]:
    """
    统一接口：在行为策略采样数据（data_behavior）中各episode的目标（desired_goal）上进行rollout，
    记录每个episode的结果，并输出平均回报（return）。
    
    根据 env_id 自动路由到对应的评估函数：
    - MyReach* -> evaluate_my_reach
    - MyPush* -> evaluate_my_push_slide
    - MySlide* -> evaluate_my_push_slide
    - FlyCraft* -> evaluate_flycraft
    
    Args:
        eval_pi_ckpt_path: 评估策略的checkpoint路径
        data_behavior: 行为策略采样的数据集
        behavior_pi_name: 行为策略的名称（用于保存结果文件）
        algo_type: 算法类型（ppo, sac, her等）
        env_id: 环境ID
        env_cfg: 环境配置（DictConfig），如果提供，会自动提取flycraft所需的config_file和custom_config
        env_config_path: flycraft环境的配置文件路径（如果env_cfg未提供，需要手动指定）
        env_custom_config: flycraft环境的自定义配置（如果env_cfg未提供，需要手动指定）
        process_num: 并行进程数
        gamma: 折扣因子
        seed: 随机种子
        
    Returns:
        tuple[float, float]: (平均回报, 标准差)
    """
    # 确保 eval_pi_ckpt_path 是 Path 对象
    if isinstance(eval_pi_ckpt_path, str):
        eval_pi_ckpt_path = Path(eval_pi_ckpt_path)
    
    # 根据 env_id 路由到对应的评估函数
    if env_cfg.env_id.startswith("FlyCraft"):
        return evaluate_flycraft(
            eval_pi_ckpt_path=eval_pi_ckpt_path,
            data_behavior=data_behavior,
            behavior_pi_name=behavior_pi_name,
            algo_type=algo_type,
            env_cfg=env_cfg,
            process_num=process_num,
            gamma=gamma,
            seed=seed,
            logger=logger
        )
    
    elif env_cfg.env_id.startswith("MyReach"):
        # MyReach 环境
        return evaluate_my_reach(
            eval_pi_ckpt_path=eval_pi_ckpt_path,
            data_behavior=data_behavior,
            behavior_pi_name=behavior_pi_name,
            algo_type=algo_type,
            env_id=env_cfg.env_id,
            process_num=process_num,
            gamma=gamma,
            seed=seed,
            logger=logger
        )

    elif env_cfg.env_id.startswith("MyPush") or env_cfg.env_id.startswith("MySlide"):
        # MyPush 和 MySlide 环境使用相同的评估函数
        return evaluate_my_push_slide(
            eval_pi_ckpt_path=eval_pi_ckpt_path,
            data_behavior=data_behavior,
            behavior_pi_name=behavior_pi_name,
            algo_type=algo_type,
            env_id=env_cfg.env_id,
            process_num=process_num,
            gamma=gamma,
            seed=seed,
            logger=logger
        )
    
    else:
        raise ValueError(
            f"Unsupported environment: {env_cfg.env_id}. "
            f"Supported environments: FlyCraft*, MyReach*, MyPush*, MySlide*"
        )
