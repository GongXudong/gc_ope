#!/bin/bash

uv run scripts/train.py experiment_name=easy/ppo/seed_1 algo=ppo callback=ppo env=flycraft_for_ppo evaluate=flycraft train_steps=2e8 algo.seed=5 env.train_env.seed=2 env.evaluation_env.seed=8 env.callback_env.seed=9
uv run scripts/train.py experiment_name=easy/ppo/seed_2 algo=ppo callback=ppo env=flycraft_for_ppo evaluate=flycraft train_steps=2e8 algo.seed=14 env.train_env.seed=12 env.evaluation_env.seed=11 env.callback_env.seed=19
uv run scripts/train.py experiment_name=easy/ppo/seed_3 algo=ppo callback=ppo env=flycraft_for_ppo evaluate=flycraft train_steps=2e8 algo.seed=26 env.train_env.seed=21 env.evaluation_env.seed=23 env.callback_env.seed=27
uv run scripts/train.py experiment_name=easy/ppo/seed_4 algo=ppo callback=ppo env=flycraft_for_ppo evaluate=flycraft train_steps=2e8 algo.seed=34 env.train_env.seed=32 env.evaluation_env.seed=36 env.callback_env.seed=35
uv run scripts/train.py experiment_name=easy/ppo/seed_5 algo=ppo callback=ppo env=flycraft_for_ppo evaluate=flycraft train_steps=2e8 algo.seed=49 env.train_env.seed=43 env.evaluation_env.seed=48 env.callback_env.seed=42
