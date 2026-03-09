#!/bin/bash

## generate desired goals randomly
uv run --offline scripts/evaluate_new/evaluate_my_push_slide.py algo_type=sac ckpt_dir=checkpoints/my_push/sac/seed_1 env=my_push process_num=16 eval_cfg=my_push_random_dg_set eval_cfg.dg_gen_method=random eval_cfg.eval_dg_num=1000 gamma=0.95 eval_res_csv_file_suffix=eval_res_on_rb seed=12
uv run --offline scripts/evaluate_new/evaluate_my_push_slide.py algo_type=sac ckpt_dir=checkpoints/my_push/sac/seed_2 env=my_push process_num=16 eval_cfg=my_push_random_dg_set eval_cfg.dg_gen_method=random eval_cfg.eval_dg_num=1000 gamma=0.95 eval_res_csv_file_suffix=eval_res_on_rb seed=25
uv run --offline scripts/evaluate_new/evaluate_my_push_slide.py algo_type=sac ckpt_dir=checkpoints/my_push/sac/seed_3 env=my_push process_num=16 eval_cfg=my_push_random_dg_set eval_cfg.dg_gen_method=random eval_cfg.eval_dg_num=1000 gamma=0.95 eval_res_csv_file_suffix=eval_res_on_rb seed=39
uv run --offline scripts/evaluate_new/evaluate_my_push_slide.py algo_type=sac ckpt_dir=checkpoints/my_push/sac/seed_4 env=my_push process_num=16 eval_cfg=my_push_random_dg_set eval_cfg.dg_gen_method=random eval_cfg.eval_dg_num=1000 gamma=0.95 eval_res_csv_file_suffix=eval_res_on_rb seed=47
uv run --offline scripts/evaluate_new/evaluate_my_push_slide.py algo_type=sac ckpt_dir=checkpoints/my_push/sac/seed_5 env=my_push process_num=16 eval_cfg=my_push_random_dg_set eval_cfg.dg_gen_method=random eval_cfg.eval_dg_num=1000 gamma=0.95 eval_res_csv_file_suffix=eval_res_on_rb seed=51

## omega_dscnt_0_9_b_0_n_100_eval_96_seed_*
suffix=omega_dscnt_0_9_b_0_n_100_eval_96_seed_
uv run --offline scripts/evaluate_new/evaluate_my_push_slide.py algo_type=sac ckpt_dir=checkpoints/my_push/sac/${suffix}1 env=my_push process_num=16 eval_cfg=my_push_random_dg_set eval_cfg.dg_gen_method=random eval_cfg.eval_dg_num=1000 gamma=0.95 eval_res_csv_file_suffix=eval_res_on_rb seed=21
uv run --offline scripts/evaluate_new/evaluate_my_push_slide.py algo_type=sac ckpt_dir=checkpoints/my_push/sac/${suffix}2 env=my_push process_num=16 eval_cfg=my_push_random_dg_set eval_cfg.dg_gen_method=random eval_cfg.eval_dg_num=1000 gamma=0.95 eval_res_csv_file_suffix=eval_res_on_rb seed=52
uv run --offline scripts/evaluate_new/evaluate_my_push_slide.py algo_type=sac ckpt_dir=checkpoints/my_push/sac/${suffix}3 env=my_push process_num=16 eval_cfg=my_push_random_dg_set eval_cfg.dg_gen_method=random eval_cfg.eval_dg_num=1000 gamma=0.95 eval_res_csv_file_suffix=eval_res_on_rb seed=93
uv run --offline scripts/evaluate_new/evaluate_my_push_slide.py algo_type=sac ckpt_dir=checkpoints/my_push/sac/${suffix}4 env=my_push process_num=16 eval_cfg=my_push_random_dg_set eval_cfg.dg_gen_method=random eval_cfg.eval_dg_num=1000 gamma=0.95 eval_res_csv_file_suffix=eval_res_on_rb seed=41
uv run --offline scripts/evaluate_new/evaluate_my_push_slide.py algo_type=sac ckpt_dir=checkpoints/my_push/sac/${suffix}5 env=my_push process_num=16 eval_cfg=my_push_random_dg_set eval_cfg.dg_gen_method=random eval_cfg.eval_dg_num=1000 gamma=0.95 eval_res_csv_file_suffix=eval_res_on_rb seed=23

## omega_rb_dscnt_0_9_b_0_n_100_eval_96_seed_*
suffix=omega_rb_dscnt_0_9_b_0_n_100_eval_96_seed_
uv run --offline scripts/evaluate_new/evaluate_my_push_slide.py algo_type=sac ckpt_dir=checkpoints/my_push/sac/${suffix}1 env=my_push process_num=16 eval_cfg=my_push_random_dg_set eval_cfg.dg_gen_method=random eval_cfg.eval_dg_num=1000 gamma=0.95 eval_res_csv_file_suffix=eval_res_on_rb seed=56
uv run --offline scripts/evaluate_new/evaluate_my_push_slide.py algo_type=sac ckpt_dir=checkpoints/my_push/sac/${suffix}2 env=my_push process_num=16 eval_cfg=my_push_random_dg_set eval_cfg.dg_gen_method=random eval_cfg.eval_dg_num=1000 gamma=0.95 eval_res_csv_file_suffix=eval_res_on_rb seed=28
uv run --offline scripts/evaluate_new/evaluate_my_push_slide.py algo_type=sac ckpt_dir=checkpoints/my_push/sac/${suffix}3 env=my_push process_num=16 eval_cfg=my_push_random_dg_set eval_cfg.dg_gen_method=random eval_cfg.eval_dg_num=1000 gamma=0.95 eval_res_csv_file_suffix=eval_res_on_rb seed=82
uv run --offline scripts/evaluate_new/evaluate_my_push_slide.py algo_type=sac ckpt_dir=checkpoints/my_push/sac/${suffix}4 env=my_push process_num=16 eval_cfg=my_push_random_dg_set eval_cfg.dg_gen_method=random eval_cfg.eval_dg_num=1000 gamma=0.95 eval_res_csv_file_suffix=eval_res_on_rb seed=16
uv run --offline scripts/evaluate_new/evaluate_my_push_slide.py algo_type=sac ckpt_dir=checkpoints/my_push/sac/${suffix}5 env=my_push process_num=16 eval_cfg=my_push_random_dg_set eval_cfg.dg_gen_method=random eval_cfg.eval_dg_num=1000 gamma=0.95 eval_res_csv_file_suffix=eval_res_on_rb seed=18
