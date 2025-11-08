#!/bin/bash

uv run scripts/train.py experiment_name=easy/sac/seed_1 algo=sac_for_my_reach callback=sac env=my_reach_for_sac evaluate=my_reach train_steps=1e6 algo.seed=5 env.train_env.seed=2 env.evaluation_env.seed=8 env.callback_env.seed=9
