#!/bin/bash

## generate desired goals randomly
uv run scripts/evaluate_ckpt/evaluate_flycraft.py algo_type=ppo ckpt_dir=checkpoints/flycraft/easy/ppo/seed_1 env=flycraft_for_ppo process_num=64 eval_cfg.dg_gen_method=random eval_cfg.eval_dg_num=1000 eval_res_csv_file_suffix=eval_res_on_random seed=23
uv run scripts/evaluate_ckpt/evaluate_flycraft.py algo_type=ppo ckpt_dir=checkpoints/flycraft/easy/ppo/seed_2 env=flycraft_for_ppo process_num=64 eval_cfg.dg_gen_method=random eval_cfg.eval_dg_num=1000 eval_res_csv_file_suffix=eval_res_on_random seed=31
uv run scripts/evaluate_ckpt/evaluate_flycraft.py algo_type=ppo ckpt_dir=checkpoints/flycraft/easy/ppo/seed_3 env=flycraft_for_ppo process_num=64 eval_cfg.dg_gen_method=random eval_cfg.eval_dg_num=1000 eval_res_csv_file_suffix=eval_res_on_random seed=45
uv run scripts/evaluate_ckpt/evaluate_flycraft.py algo_type=ppo ckpt_dir=checkpoints/flycraft/easy/ppo/seed_4 env=flycraft_for_ppo process_num=64 eval_cfg.dg_gen_method=random eval_cfg.eval_dg_num=1000 eval_res_csv_file_suffix=eval_res_on_random seed=51
uv run scripts/evaluate_ckpt/evaluate_flycraft.py algo_type=ppo ckpt_dir=checkpoints/flycraft/easy/ppo/seed_5 env=flycraft_for_ppo process_num=64 eval_cfg.dg_gen_method=random eval_cfg.eval_dg_num=1000 eval_res_csv_file_suffix=eval_res_on_random seed=64

## generate desired goals by enumerating over fixed dg intervals
uv run scripts/evaluate_ckpt/evaluate_flycraft.py algo_type=ppo ckpt_dir=checkpoints/flycraft/easy/ppo/seed_1 env=flycraft_for_ppo process_num=64 eval_cfg.dg_gen_method=fixed eval_cfg.v_interval=10 eval_cfg.mu_interval=2 eval_cfg.chi_interval=2 eval_res_csv_file_suffix=eval_res_on_fixed seed=11
uv run scripts/evaluate_ckpt/evaluate_flycraft.py algo_type=ppo ckpt_dir=checkpoints/flycraft/easy/ppo/seed_2 env=flycraft_for_ppo process_num=64 eval_cfg.dg_gen_method=fixed eval_cfg.v_interval=10 eval_cfg.mu_interval=2 eval_cfg.chi_interval=2 eval_res_csv_file_suffix=eval_res_on_fixed seed=25
uv run scripts/evaluate_ckpt/evaluate_flycraft.py algo_type=ppo ckpt_dir=checkpoints/flycraft/easy/ppo/seed_3 env=flycraft_for_ppo process_num=64 eval_cfg.dg_gen_method=fixed eval_cfg.v_interval=10 eval_cfg.mu_interval=2 eval_cfg.chi_interval=2 eval_res_csv_file_suffix=eval_res_on_fixed seed=32
uv run scripts/evaluate_ckpt/evaluate_flycraft.py algo_type=ppo ckpt_dir=checkpoints/flycraft/easy/ppo/seed_4 env=flycraft_for_ppo process_num=64 eval_cfg.dg_gen_method=fixed eval_cfg.v_interval=10 eval_cfg.mu_interval=2 eval_cfg.chi_interval=2 eval_res_csv_file_suffix=eval_res_on_fixed seed=49
uv run scripts/evaluate_ckpt/evaluate_flycraft.py algo_type=ppo ckpt_dir=checkpoints/flycraft/easy/ppo/seed_5 env=flycraft_for_ppo process_num=64 eval_cfg.dg_gen_method=fixed eval_cfg.v_interval=10 eval_cfg.mu_interval=2 eval_cfg.chi_interval=2 eval_res_csv_file_suffix=eval_res_on_fixed seed=57
