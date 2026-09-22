# 新对比方法：直接加权 EM GMM

方法名 `gmm_em`，Fig.6 图例为 `GMM (weighted EM)`。旧方法 `gmm` 及其重采样实现、
参数、CSV 和图均保留；新方法是独立对照，不覆盖或冒充原 GMM。

## 优化的 loss

本机 sklearn 1.7.2 的 `GaussianMixture.fit` 签名是 `(X, y=None)`，没有
`sample_weight`。[官方接口文档](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture.fit)
也列出了这一接口。新实现不依赖 sklearn 私有 GMM 函数，直接优化：

\[
L(\theta)=-\frac{\sum_i w_i\log p_\theta(x_i)}{\sum_i w_i},\qquad
p_\theta(x)=\sum_k\pi_k\mathcal N(x;\mu_k,\Sigma_k).
\]

即用户明确要求的**加权平均负对数似然**。EM 是求解这个目标的方法，权重确实
改变参数更新，不是只给最后汇报的 loss 乘权重。

E-step：

\[
r_{ik}=\frac{\pi_k\mathcal N(x_i;\mu_k,\Sigma_k)}
{\sum_j\pi_j\mathcal N(x_i;\mu_j,\Sigma_j)}.
\]

条件责任度仍按普通 GMM 计算；同一个样本的权重在条件概率归一化中会抵消。
M-step 用加权充分统计量：

\[
N_k=\sum_iw_ir_{ik},\quad
\pi_k=\frac{N_k}{\sum_iw_i},\quad
\mu_k=\frac{\sum_iw_ir_{ik}x_i}{N_k},\quad
\Sigma_k=\frac{\sum_iw_ir_{ik}(x_i-\mu_k)(x_i-\mu_k)^T}{N_k}+\lambda I.
\]

`diag` 只保留对角方差。`reg_covar` 是防止协方差退化的数值正则，沿用旧值
1e-6；有正则时不声称未正则化似然每一步严格上升。单调性测试在正定数据、正则为0时执行。

## 实现与兼容性

- 数学模型：`src/gc_ope/evaluate/utils/weighted_gmm.py`，支持 full/diag、多个初始化、
  log-space 责任度、采样、密度、收敛诊断、零权重排除。
- 项目接口：`src/gc_ope/evaluate/evaluator_gmm_em.py`，共用原容器、原坐标密度和
  课程学习接口。标准化仍为成功目标上的不加权 StandardScaler，未另改为加权 scaler。
- 初始化使用支持 sample_weight 的 KMeans；其后直接对全部成功样本做加权 EM，
  没有重采样1000条的步骤。少样本时分量数上限取不同的正权重成功坐标数。
- 各次初始化按加权平均对数似然选择最佳结果；停止依据也使用这个加权目标。
  `weighted_nll` 记录最终 loss，`lower_bounds` 保留相反符号的目标曲线。
- 达到迭代上限仍未收敛会记录 warning；计算成功不等于拟合准确，更不保证 KL 更小。

历史侧每轮 fixed 全部记录抽100条，包含当前轮，成功记录按
`0.9 ** ((当前步数 - 来源步数) / 10000)` 加权。参考侧读取当前全量成功目标，
权重全1，使用**同种新方法、同配置、独立实例**拟合。MC仍为10000点×5次、参考||历史。

新配置 `configs/evaluate/push_gmm_em_all100.json`：5分量、full、n_init=1、max_iter=200、
tol=1e-3、reg_covar=1e-6、random_state=0。除去原重采样数量，尽量保持与旧GMM一致。
两个版本的初始化样本及局部最优仍可能不同，因此不能把所有数值变化都归因于权重近似误差。

## 验证记录

数学测试覆盖 full/diag：单高斯解析加权均值/协方差、整数权重与重复记录对照、
等权与 sklearn 对照（使用可稳定分离的数据）、权重整体缩放不变、零权重不影响结果、
加权目标单调性、采样矩、密度一致性、随机流隔离、奇异数据及停止条件。
另外通过旧 GMM 回归、连续密度/MC、两侧同类独立拟合、真实 Push/SAC 短训练、
五 seed 调度与第六条曲线来源检查。没有运行整个 tests/。

真实数据检查：五 seed × 10000、100000、400000、1000000 步，共20项，
MC均为10000×5。19有效、1项因seed2/10000历史无成功样本跳过，0错误、0收敛提醒。
输出 `logs/gmm_em_validation_20260922/`，逐点新旧对照 `comparison.csv`。

| checkpoint | 原重采样 GMM 平均KL | 直接加权 EM 平均KL |
| --- | ---: | ---: |
| 10000（4个有效seed） | 976926.610247 | 996817.608547 |
| 100000 | 3.458273 | 4.145521 |
| 400000 | 0.342535 | 0.323577 |
| 1000000 | 0.083926 | 0.103426 |

早期成功点极少，两种GMM都有巨大的KL，原值保留。此检查证明流程可用，
**没有证明新方法更好**。未根据参考侧KL调参，未启动新的500项全量实验。

## 全量与六方法绘图命令

新增方法可单独运行，仍为五 seed 并行、每 seed 四 worker：

```bash
conda run --no-capture-output -n gc_ope python \
  /home/tacmon/workspace/ex_SSD/gc_ope_refactor/scripts/run_push_all100.py \
  --methods gmm_em \
  --config /home/tacmon/workspace/ex_SSD/gc_ope_refactor/configs/evaluate/push_gmm_em_all100.json \
  --output /home/tacmon/workspace/ex_SSD/gc_ope_refactor/logs/gmm_em_all100_5x4
```

完成后新增第六条曲线，原重采样GMM仍显示为 `GMM`：

```bash
conda run --no-capture-output -n gc_ope python \
  /home/tacmon/workspace/ex_SSD/gc_ope_refactor/scripts/plot_fig6.py \
  --nn-result-root /home/tacmon/workspace/ex_SSD/gc_ope_refactor/logs/nn_logloss_v2_all100_5x4 \
  --gmm-em-result-root /home/tacmon/workspace/ex_SSD/gc_ope_refactor/logs/gmm_em_all100_5x4 \
  --output /home/tacmon/workspace/ex_SSD/gc_ope_refactor/plots/fig6_with_gmm_em
```

不传新参数时，原五方法绘图行为不变。课程配置可直接使用
`estimator_config: {method: gmm_em, parameters: {n_components: 5, covariance_type: full}}`。
