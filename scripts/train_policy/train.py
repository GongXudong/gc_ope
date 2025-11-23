"""
1.获得env(train, evaluate, callback)
2.获得algo
3.准备callbacks
4.训练
5.评估
"""

from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

from stable_baselines3.common.logger import configure, Logger

from gc_ope.env.get_vec_env import get_vec_env
from gc_ope.algorithm.get_algorithm import get_algo
from gc_ope.algorithm.get_callbacks import get_callback_list
from gc_ope.algorithm.get_evaluate_method import get_evaluate_method


PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent


@hydra.main(version_base=None, config_path="../../configs/train", config_name="config")
def train_agent(cfg: DictConfig) -> None:

    # 0.配置logger
    sb3_logger: Logger = configure(
        folder=str((PROJECT_ROOT_DIR / cfg.log_path).absolute()),
        format_strings=['stdout', 'log', 'csv', 'tensorboard'],
    )

    # 1.set vec_env
    train_env, eval_env, callback_env = get_vec_env(env_cfg=cfg.env)
    sb3_logger.info(f"Init env successfully, {str(train_env)}")

    # 2.get algorithm
    algo = get_algo(
        algo_cfg=cfg.algo,
        env=train_env,
    )
    sb3_logger.info(f"Init policy successfully, {str(algo.policy)}")
    algo.set_logger(sb3_logger)

    # 3.get callbacks
    callback_list = get_callback_list(
        callback_cfg=cfg.callback,
        env_cfg=cfg.env,
        env=callback_env,
    )
    sb3_logger.info(f"Init callback successfully, {str(callback_list)}")

    # 4.train
    algo.learn(
        total_timesteps=int(cfg.train_steps),
        callback=callback_list,
    )

    # 5.evaluate
    evaluate_method = get_evaluate_method(
        eval_cfg=cfg.evaluate,
        env_cfg=cfg.env,
    )
    res = evaluate_method(
        model=algo.policy,
        env=eval_env,
        n_eval_episodes=int(cfg.evaluate.evaluate_nums_in_evaluation * cfg.env.evaluation_env.num_process),
    )
    sb3_logger.info(f"Evaluation results after training: {res}.")

# 查看配置的值：uv run scripts/train.py experiment_name=test --cfg job
# 查看解析后的配置值（注意：需要先提供placeholder的值）：uv run scripts/train.py experiment_name=test --resolve --cfg job
if __name__ == "__main__":
    train_agent()
