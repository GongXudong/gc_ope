#!/bin/bash

## generate desired goals randomly
uv run scripts/evaluate_flycraft.py algo_type=sac ckpt_dir=checkpoints/flycraft/easy/her/seed_1 env=flycraft_for_sac num_process=64 eval_cfg.dg_gen_method=random eval_cfg.eval_dg_num=1000 seed=23
uv run scripts/evaluate_flycraft.py algo_type=sac ckpt_dir=checkpoints/flycraft/easy/her/seed_2 env=flycraft_for_sac num_process=64 eval_cfg.dg_gen_method=random eval_cfg.eval_dg_num=1000 seed=31
uv run scripts/evaluate_flycraft.py algo_type=sac ckpt_dir=checkpoints/flycraft/easy/her/seed_3 env=flycraft_for_sac num_process=64 eval_cfg.dg_gen_method=random eval_cfg.eval_dg_num=1000 seed=45
uv run scripts/evaluate_flycraft.py algo_type=sac ckpt_dir=checkpoints/flycraft/easy/her/seed_4 env=flycraft_for_sac num_process=64 eval_cfg.dg_gen_method=random eval_cfg.eval_dg_num=1000 seed=51
uv run scripts/evaluate_flycraft.py algo_type=sac ckpt_dir=checkpoints/flycraft/easy/her/seed_5 env=flycraft_for_sac num_process=64 eval_cfg.dg_gen_method=random eval_cfg.eval_dg_num=1000 seed=64

## generate desired goals by enumerating over fixed dg intervals
uv run scripts/evaluate_flycraft.py algo_type=sac ckpt_dir=checkpoints/flycraft/easy/her/seed_1 env=flycraft_for_sac num_process=64 eval_cfg.dg_gen_method=fixed eval_cfg.v_interval=10 eval_cfg.mu_interval=2 eval_cfg.chi_interval=2 seed=11
uv run scripts/evaluate_flycraft.py algo_type=sac ckpt_dir=checkpoints/flycraft/easy/her/seed_2 env=flycraft_for_sac num_process=64 eval_cfg.dg_gen_method=fixed eval_cfg.v_interval=10 eval_cfg.mu_interval=2 eval_cfg.chi_interval=2 seed=25
uv run scripts/evaluate_flycraft.py algo_type=sac ckpt_dir=checkpoints/flycraft/easy/her/seed_3 env=flycraft_for_sac num_process=64 eval_cfg.dg_gen_method=fixed eval_cfg.v_interval=10 eval_cfg.mu_interval=2 eval_cfg.chi_interval=2 seed=32
uv run scripts/evaluate_flycraft.py algo_type=sac ckpt_dir=checkpoints/flycraft/easy/her/seed_4 env=flycraft_for_sac num_process=64 eval_cfg.dg_gen_method=fixed eval_cfg.v_interval=10 eval_cfg.mu_interval=2 eval_cfg.chi_interval=2 seed=49
uv run scripts/evaluate_flycraft.py algo_type=sac ckpt_dir=checkpoints/flycraft/easy/her/seed_5 env=flycraft_for_sac num_process=64 eval_cfg.dg_gen_method=fixed eval_cfg.v_interval=10 eval_cfg.mu_interval=2 eval_cfg.chi_interval=2 seed=57
