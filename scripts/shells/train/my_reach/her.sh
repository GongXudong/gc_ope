#!/bin/bash

uv run scripts/train.py experiment_name=easy/her/seed_1 algo=her_for_my_reach callback=sac env=my_reach_for_sac evaluate=my_reach train_steps=1e6 algo.seed=5 env.train_env.seed=2 env.evaluation_env.seed=8 env.callback_env.seed=9
uv run scripts/train.py experiment_name=easy/her/seed_2 algo=her_for_my_reach callback=sac env=my_reach_for_sac evaluate=my_reach train_steps=1e6 algo.seed=11 env.train_env.seed=15 env.evaluation_env.seed=18 env.callback_env.seed=13
uv run scripts/train.py experiment_name=easy/her/seed_3 algo=her_for_my_reach callback=sac env=my_reach_for_sac evaluate=my_reach train_steps=1e6 algo.seed=25 env.train_env.seed=22 env.evaluation_env.seed=27 env.callback_env.seed=26
uv run scripts/train.py experiment_name=easy/her/seed_4 algo=her_for_my_reach callback=sac env=my_reach_for_sac evaluate=my_reach train_steps=1e6 algo.seed=32 env.train_env.seed=35 env.evaluation_env.seed=39 env.callback_env.seed=34
uv run scripts/train.py experiment_name=easy/her/seed_5 algo=her_for_my_reach callback=sac env=my_reach_for_sac evaluate=my_reach train_steps=1e6 algo.seed=48 env.train_env.seed=41 env.evaluation_env.seed=47 env.callback_env.seed=46
