#!/usr/bin/env bash
# 五次独立 Push/SAC/OMEGA-PE 训练，只把能力估计器替换为最终版 GMM。
# 与旧脚本一样逐 seed 串行；每次训练内部保留 16 个 callback 评估环境。
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$repo_root"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# 子进程中的旧 wrapper 用 print 记录课程；关闭缓冲，避免短运行/中断时遗漏。
export PYTHONUNBUFFERED=1

dry_run=false
selected_seed=all
while (($#)); do
    case "$1" in
        --dry-run) dry_run=true; shift ;;
        --seed)
            if (($# < 2)) || [[ ! "$2" =~ ^[1-5]$ ]]; then
                echo "--seed 必须是 1 到 5。" >&2; exit 2
            fi
            selected_seed="$2"; shift 2 ;;
        *) echo "用法：$0 [--dry-run] [--seed 1到5]" >&2; exit 2 ;;
    esac
done

run_seed() {
    local seed="$1"
    shift
    [[ "$selected_seed" == all || "$selected_seed" == "$seed" ]] || return 0
    printf '即将运行 seed %s：' "$seed"
    printf ' %q' "$@"
    printf '\n'
    "$dry_run" && return 0

    local name="sac/omega_gmm_dscnt_0_9_b_0_n_100_eval_96_seed_${seed}"
    local log_dir="logs/my_push/$name"
    local ckpt_dir="checkpoints/my_push/$name"
    # 沿用师兄课程图 notebook 的目录和文件命名，仅增加 gmm 方法名。
    local process_log="logs_in_process/my_push/sac/my_push_${name//\//_}.txt"
    # 原子创建运行目录：阻止同 seed 重复启动，也避免从头训练覆盖已有结果。
    # 中断后需要另取 experiment_name 手动启动；此入口不伪装成训练续跑。
    if [[ -e "$ckpt_dir" || -e "$process_log" ]]; then
        echo "已有 checkpoint 或课程文本日志，停止：$name" >&2; return 1
    fi
    mkdir -p "$(dirname "$log_dir")"
    if ! mkdir "$log_dir"; then
        echo "已有日志或同 seed 正在运行，停止：$log_dir" >&2; return 1
    fi
    mkdir -p "$(dirname "$process_log")"
    # Hydra 配置也随此次训练保存；pipefail 保留训练失败的退出码。
    "$@" "hydra.run.dir=$log_dir/hydra" 2>&1 | conda run --no-capture-output -n gc_ope tee "$process_log" "$log_dir/console.log"
}

# 以下四组种子对应旧 sac_omega.sh 的前五个正式实验，保持逐项可读。
run_seed 1 conda run --no-capture-output -n gc_ope python scripts/train_policy/train.py experiment_name=sac/omega_gmm_dscnt_0_9_b_0_n_100_eval_96_seed_1 algo=sac_for_my_push algo.seed=5 env=my_push_omega_gmm env.train_env.seed=2 env.evaluation_env.seed=8 env.callback_env.seed=9 callback=sac_omega callback.0.evaluate_nums_in_callback=6 env.callback_env.num_process=16 evaluate=my_push train_steps=1e6
run_seed 2 conda run --no-capture-output -n gc_ope python scripts/train_policy/train.py experiment_name=sac/omega_gmm_dscnt_0_9_b_0_n_100_eval_96_seed_2 algo=sac_for_my_push algo.seed=12 env=my_push_omega_gmm env.train_env.seed=15 env.evaluation_env.seed=14 env.callback_env.seed=17 callback=sac_omega callback.0.evaluate_nums_in_callback=6 env.callback_env.num_process=16 evaluate=my_push train_steps=1e6
run_seed 3 conda run --no-capture-output -n gc_ope python scripts/train_policy/train.py experiment_name=sac/omega_gmm_dscnt_0_9_b_0_n_100_eval_96_seed_3 algo=sac_for_my_push algo.seed=26 env=my_push_omega_gmm env.train_env.seed=27 env.evaluation_env.seed=23 env.callback_env.seed=29 callback=sac_omega callback.0.evaluate_nums_in_callback=6 env.callback_env.num_process=16 evaluate=my_push train_steps=1e6
run_seed 4 conda run --no-capture-output -n gc_ope python scripts/train_policy/train.py experiment_name=sac/omega_gmm_dscnt_0_9_b_0_n_100_eval_96_seed_4 algo=sac_for_my_push algo.seed=37 env=my_push_omega_gmm env.train_env.seed=31 env.evaluation_env.seed=36 env.callback_env.seed=33 callback=sac_omega callback.0.evaluate_nums_in_callback=6 env.callback_env.num_process=16 evaluate=my_push train_steps=1e6
run_seed 5 conda run --no-capture-output -n gc_ope python scripts/train_policy/train.py experiment_name=sac/omega_gmm_dscnt_0_9_b_0_n_100_eval_96_seed_5 algo=sac_for_my_push algo.seed=43 env=my_push_omega_gmm env.train_env.seed=46 env.evaluation_env.seed=49 env.callback_env.seed=44 callback=sac_omega callback.0.evaluate_nums_in_callback=6 env.callback_env.num_process=16 evaluate=my_push train_steps=1e6
