#!/bin/bash

uv run scripts/train.py experiment_name=easy/her/seed_1 algo=her_for_flycraft callback=sac env=flycraft_for_sac evaluate=flycraft train_steps=1e6 algo.seed=5 env.train_env.seed=2 env.evaluation_env.seed=8 env.callback_env.seed=9
uv run scripts/train.py experiment_name=easy/her/seed_2 algo=her_for_flycraft callback=sac env=flycraft_for_sac evaluate=flycraft train_steps=1e6 algo.seed=14 env.train_env.seed=12 env.evaluation_env.seed=11 env.callback_env.seed=19
uv run scripts/train.py experiment_name=easy/her/seed_3 algo=her_for_flycraft callback=sac env=flycraft_for_sac evaluate=flycraft train_steps=1e6 algo.seed=26 env.train_env.seed=21 env.evaluation_env.seed=23 env.callback_env.seed=27
uv run scripts/train.py experiment_name=easy/her/seed_4 algo=her_for_flycraft callback=sac env=flycraft_for_sac evaluate=flycraft train_steps=1e6 algo.seed=34 env.train_env.seed=32 env.evaluation_env.seed=36 env.callback_env.seed=35
uv run scripts/train.py experiment_name=easy/her/seed_5 algo=her_for_flycraft callback=sac env=flycraft_for_sac evaluate=flycraft train_steps=1e6 algo.seed=49 env.train_env.seed=43 env.evaluation_env.seed=48 env.callback_env.seed=42
