#!/bin/bash

## generate desired goals randomly
uv run --offline scripts/evaluate_new/evaluate_my_push_slide.py algo_type=sac ckpt_dir=checkpoints/my_slide/sac/seed_1 env=my_slide process_num=32 eval_cfg=my_slide_random_dg_set eval_cfg.dg_gen_method=random eval_cfg.eval_dg_num=1000 gamma=0.95 eval_res_csv_file_suffix=eval_res_on_rb seed=12
uv run --offline scripts/evaluate_new/evaluate_my_push_slide.py algo_type=sac ckpt_dir=checkpoints/my_slide/sac/seed_2 env=my_slide process_num=32 eval_cfg=my_slide_random_dg_set eval_cfg.dg_gen_method=random eval_cfg.eval_dg_num=1000 gamma=0.95 eval_res_csv_file_suffix=eval_res_on_rb seed=25
uv run --offline scripts/evaluate_new/evaluate_my_push_slide.py algo_type=sac ckpt_dir=checkpoints/my_slide/sac/seed_3 env=my_slide process_num=32 eval_cfg=my_slide_random_dg_set eval_cfg.dg_gen_method=random eval_cfg.eval_dg_num=1000 gamma=0.95 eval_res_csv_file_suffix=eval_res_on_rb seed=39
uv run --offline scripts/evaluate_new/evaluate_my_push_slide.py algo_type=sac ckpt_dir=checkpoints/my_slide/sac/seed_4 env=my_slide process_num=32 eval_cfg=my_slide_random_dg_set eval_cfg.dg_gen_method=random eval_cfg.eval_dg_num=1000 gamma=0.95 eval_res_csv_file_suffix=eval_res_on_rb seed=47
uv run --offline scripts/evaluate_new/evaluate_my_push_slide.py algo_type=sac ckpt_dir=checkpoints/my_slide/sac/seed_5 env=my_slide process_num=32 eval_cfg=my_slide_random_dg_set eval_cfg.dg_gen_method=random eval_cfg.eval_dg_num=1000 gamma=0.95 eval_res_csv_file_suffix=eval_res_on_rb seed=51
