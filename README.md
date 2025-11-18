# Goal-Conditioned Off Policy Evaluation

## Prepare python Environment

To run the **Reach** task, ensure that **pybullet** is installed properly. The installation of pybullet depends on **gcc**. If compilation-related errors occur on Linux systems, check if gcc is installed correctly.

```bash
sudo apt update
sudo apt install build-essential python3-dev libssl-dev libffi-dev libxml2 libxml2-dev libxslt1-dev zlib1g-dev
sudo apt install gcc
```

## Train policies

```bash
# train PPO on FlyCraft-v0, refer to scripts/shells/train/flycraft/ppo.sh
uv run scripts/train.py experiment_name=easy/ppo/seed_1 algo=ppo callback=ppo env=flycraft_for_ppo evaluate=flycraft train_steps=2e8 algo.seed=5 env.train_env.seed=2 env.evaluation_env.seed=8 env.callback_env.seed=9

# train SAC on FlyCraft-v0, refer to scripts/shells/train/flycraft/sac.sh
uv run scripts/train.py experiment_name=easy/sac/seed_1 algo=sac callback=sac env=flycraft_for_sac evaluate=flycraft train_steps=1e6 algo.seed=5 env.train_env.seed=2 env.evaluation_env.seed=8 env.callback_env.seed=9

# train HER on FlyCraft-v0, refer to scripts/shells/train/flycraft/her.sh
uv run scripts/train.py experiment_name=easy/her/seed_1 algo=her callback=sac env=flycraft_for_sac evaluate=flycraft train_steps=1e6 algo.seed=5 env.train_env.seed=2 env.evaluation_env.seed=8 env.callback_env.seed=9

# train PPO on MyReach-v0, refer to scripts/shells/train/my_reach/ppo.sh
# train SAC on MyReach-v0, refer to scripts/shells/train/my_reach/sac.sh
# train HER on MyReach-v0, refer to scripts/shells/train/my_reach/her.sh
```

## Evaluate policies

```bash
# evaluate PPO policies on FlyCraft-v0, refer to scripts/shells/evaluate/flycraft/ppo.sh
## on random goal set
uv run scripts/evaluate_flycraft.py algo_type=ppo ckpt_dir=checkpoints/flycraft/easy/ppo/seed_1 env=flycraft_for_ppo process_num=64 eval_cfg.dg_gen_method=random eval_cfg.eval_dg_num=1000 eval_res_csv_file_suffix=eval_res_on_random seed=23
## on fixed goal set
uv run scripts/evaluate_flycraft.py algo_type=ppo ckpt_dir=checkpoints/flycraft/easy/ppo/seed_1 env=flycraft_for_ppo process_num=64 eval_cfg.dg_gen_method=fixed eval_cfg.v_interval=10 eval_cfg.mu_interval=2 eval_cfg.chi_interval=2 eval_res_csv_file_suffix=eval_res_on_fixed seed=11

# evaluate SAC policies on FlyCraft-v0, refer to scripts/shells/evaluate/flycraft/sac.sh
# evaluate HER policies on FlyCraft-v0, refer to scripts/shells/evaluate/flycraft/her.sh

# evaluate PPO policies on MyReach-v0, refer to scripts/shells/evaluate/my_reach/ppo.sh
# evaluate SAC policies on MyReach-v0, refer to scripts/shells/evaluate/my_reach/sac.sh
# evaluate HER policies on MyReach-v0, refer to scripts/shells/evaluate/my_reach/her.sh


```

## Predict goal-achievement abilities based on evaluation results

### Based on KDE

Predict based on KDE, refer to `scripts/sac_from_eval.ipynb`
