from pathlib import Path
from omegaconf import DictConfig

from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EveryNTimesteps

from gc_ope.algorithm.utils.my_eval_callback import MyEvalCallback


PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent


def get_callback_list(callback_cfg: DictConfig, env_cfg: DictConfig, env: VecEnv) -> list[BaseCallback]:

    callback_list = []

    for cfg in callback_cfg:
        if cfg.type == "MyEvalCallback":
            my_eval_callback = MyEvalCallback(
                eval_env=env,
                best_model_save_path=str((PROJECT_ROOT_DIR / cfg.best_model_save_path).absolute()),
                log_path=str((PROJECT_ROOT_DIR / cfg.log_path).absolute()), 
                eval_freq=cfg.eval_freq,
                n_eval_episodes=cfg.evaluate_nums_in_callback * env_cfg.callback_env.num_process,
                deterministic=cfg.deterministic, 
                render=False,
            )
            callback_list.append(my_eval_callback)
        elif cfg.type == "EveryNTimesteps_SaveCheckpoints":
            checkpoint_on_event = CheckpointCallback(
                save_freq=cfg.save_freq,
                save_path=str((PROJECT_ROOT_DIR / cfg.save_path).absolute()),
                save_replay_buffer=cfg.save_replay_buffer,
            )
            event_callback = EveryNTimesteps(
                n_steps=cfg.save_checkpoint_every_n_timesteps,
                callback=checkpoint_on_event,
            )
            callback_list.append(event_callback)
        else:
            raise ValueError(f"Can not process Callback type: {cfg.type}!")

    return callback_list
