#!/bin/bash

uv run scripts/train.py experiment_name=easy/ppo/seed_1 algo=ppo_for_my_reach callback=ppo callback.0.eval_freq=10000 callback.1.save_checkpoint_every_n_timesteps=10000 env=my_reach_for_ppo evaluate=my_reach train_steps=1e6 algo.seed=5 env.train_env.seed=2 env.evaluation_env.seed=8 env.callback_env.seed=9
