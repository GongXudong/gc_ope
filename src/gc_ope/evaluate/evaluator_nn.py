"""用成功率分类器及高斯核混合定义连续的能力分布。"""

from copy import deepcopy
from numbers import Integral
import numpy as np
from sklearn.base import clone
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KernelDensity

from gc_ope.evaluate.evaluator_base import EvaluatorBase
from gc_ope.evaluate.evaluation_result_container import EvaluationResultContainer
from gc_ope.evaluate.evaluator_common import uniform_grid_kl, InsufficientSamples


class GoalSuccessMLPClassifierEvaluator(EvaluatorBase):
    """先学习 P(成功|目标)，再以预测概率为权重平滑几何支持网格。

    支持网格只提供目标坐标，不提供额外的成功标签。密度为归一化的高斯
    混合，积分为 1；不能把网格上的概率除以面积后直接延伸到整个平面。
    """

    def __init__(
        self,
        evaluation_result_container_class=EvaluationResultContainer,
        evaluation_result_container_kwargs=None,
        support_goals=None,
        bandwidth=0.2,
        hidden_layer_sizes=None,
        n_epochs=500,
        lr=1e-3,
        alpha=1e-4,
        random_state=0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        min_epochs=50,
        tol=1e-4,
        hidden_width=None,
    ):
        super().__init__(evaluation_result_container_class, evaluation_result_container_kwargs or {})
        if bandwidth <= 0 or not np.isfinite(bandwidth):
            raise ValueError("核带宽必须有限且为正")
        self.support_goals = support_goals
        self.bandwidth = bandwidth
        # 旧的单宽度参数仍可读取，但不能与新的完整层结构同时指定。
        if hidden_width is not None:
            if hidden_layer_sizes is not None:
                raise ValueError("hidden_width 与 hidden_layer_sizes 不能同时指定")
            hidden_layer_sizes = [hidden_width, hidden_width]
        hidden_layer_sizes = [16, 16] if hidden_layer_sizes is None else hidden_layer_sizes
        if (not isinstance(hidden_layer_sizes, (list, tuple)) or not hidden_layer_sizes
                or any(isinstance(n, bool) or not isinstance(n, Integral) or n <= 0 for n in hidden_layer_sizes)):
            raise ValueError("hidden_layer_sizes 必须是非空的正整数列表或元组")
        for name, value in [("n_epochs", n_epochs), ("min_epochs", min_epochs),
                            ("n_iter_no_change", n_iter_no_change)]:
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} 必须是正整数")
        if not 0 < validation_fraction < 1 or not np.isfinite(tol) or tol < 0:
            raise ValueError("验证比例必须在 (0,1) 内，容差必须有限且非负")
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_epochs, self.min_epochs = n_epochs, min_epochs
        self.patience, self.tol = n_iter_no_change, tol
        self.random_state = random_state
        self.classifier = MLPClassifier(
            hidden_layer_sizes=tuple(hidden_layer_sizes),
            activation="relu", solver="adam", alpha=alpha,
            learning_rate_init=lr, max_iter=n_epochs, random_state=random_state,
            # sklearn 内置早停监控 accuracy，不适合这里的概率估计任务。
            early_stopping=False, n_iter_no_change=n_epochs + 1,
        )
        self.density = KernelDensity(bandwidth=bandwidth, kernel="gaussian")

    def fit_evaluator(self):
        """分类器使用全部标签；返回值继续遵守旧 evaluator 的四元组接口。"""
        container = self.eval_res_container
        goals = np.asarray(container.desired_goal_list, dtype=float)
        labels = np.asarray(container.success_list, dtype=bool)
        weights = np.asarray(getattr(container, "desired_goal_weights", np.ones(len(labels))))
        if goals.ndim != 2 or goals.shape[1] != 2 or len(goals) != len(labels):
            raise ValueError("NN 需要与标签对齐的二维目标")
        if not np.isfinite(goals).all() or not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("目标和权重必须有限，且权重为正")
        counts = np.bincount(labels.astype(int), minlength=2)
        if counts.min() == 0:
            raise InsufficientSamples("NN 需要成功和失败两类样本")
        if self.early_stopping:
            n_validation = int(np.ceil(len(labels) * self.validation_fraction))
            if counts.min() < 2 or min(n_validation, len(labels) - n_validation) < 2:
                raise InsufficientSamples("NN 样本不足以划分分层验证集")

        # 与原分类器一致：标准化使用所有训练目标，不只使用成功目标。
        scaled = self.scaler.fit_transform(goals)
        training = self._fit_classifier(scaled, labels.astype(int), weights)

        # 两侧使用同一几何网格；预测概率各自来自本侧独立拟合的分类器。
        support = np.asarray(self.support_goals, dtype=float)
        if support.ndim != 2 or support.shape[1] != 2 or not len(support) or not np.isfinite(support).all():
            raise ValueError("NN 需要有限的二维几何支持网格")
        self.support_goals_ = np.unique(support, axis=0)
        probability = self.predict_success_probability(self.support_goals_)
        self.support_weights_ = probability / probability.sum()
        self.density.fit(self.scaler.transform(self.support_goals_), sample_weight=self.support_weights_)
        positive = goals[labels]
        _, density = self.evaluate(positive)
        self.fit_diagnostics_ = {
            "n_samples": len(goals), "n_success": int(labels.sum()),
            "n_iter": training["epochs_run"], "bandwidth": self.bandwidth,
            "density_scheme": "预测成功率加权的归一化高斯核混合",
            **training,
        }
        return positive, scaled[labels], weights[labels], density

    def _fit_classifier(self, goals, labels, weights):
        """每轮只更新一次数据遍历；用本侧留出标签选择概率损失最低的模型。

        留出按记录分层，保持原实验的数据划分单位。历史重复抽样记录可能
        跨训练/验证集，因此验证损失是训练诊断，不作为独立泛化成绩。
        """
        self.classifier = clone(self.classifier)
        if not self.early_stopping:
            self.classifier.fit(goals, labels, sample_weight=weights)
            train, validation = np.arange(len(labels)), np.array([], dtype=int)
            epochs, best_epoch, stop_reason = self.n_epochs, self.n_epochs, "fixed_budget"
            self.validation_loss_curve_ = []
        else:
            train, validation = train_test_split(
                np.arange(len(labels)), test_size=self.validation_fraction,
                stratify=labels, random_state=self.random_state,
            )
            # 两个分区都必须包含两类；极少数成功记录不补造、不静默换算法。
            if min(len(np.unique(labels[train])), len(np.unique(labels[validation]))) < 2:
                raise InsufficientSamples("NN 分层划分后有分区缺少某一类")
            # partial_fit 使用独立、持续推进的随机流，不影响 SAC 的全局随机流。
            self.classifier.set_params(random_state=np.random.RandomState(self.random_state))
            best_loss, patience_loss, stale = np.inf, np.inf, 0
            self.validation_loss_curve_ = []
            stop_reason = "max_epochs"
            for epoch in range(1, self.n_epochs + 1):
                self.classifier.partial_fit(goals[train], labels[train], classes=[0, 1],
                                            sample_weight=weights[train])
                probability = self.classifier.predict_proba(goals[validation])[:, 1]
                loss = log_loss(labels[validation], probability, sample_weight=weights[validation], labels=[0, 1])
                if not np.isfinite(loss):
                    raise FloatingPointError("NN 验证损失不是有限值")
                self.validation_loss_curve_.append(float(loss))
                if loss < best_loss:
                    best_loss, best_epoch = loss, epoch
                    best_model = deepcopy(self.classifier)
                if loss < patience_loss - self.tol:
                    patience_loss, stale = loss, 0
                else:
                    stale += 1
                if epoch >= self.min_epochs and stale >= self.patience:
                    stop_reason = "validation_log_loss"
                    break
            epochs = epoch
            self.classifier = best_model

        probability = self.classifier.predict_proba(goals)[:, 1]
        train_loss = log_loss(labels[train], probability[train], sample_weight=weights[train], labels=[0, 1])
        # 常数基线只用训练区的成功比例，不能看验证标签后再设置它。
        prevalence = float(np.average(labels[train], weights=weights[train]))
        quality_warnings = []
        validation_loss = baseline_loss = None
        if len(validation):
            validation_loss = float(log_loss(labels[validation], probability[validation],
                                            sample_weight=weights[validation], labels=[0, 1]))
            baseline_loss = float(log_loss(labels[validation], np.full(len(validation), prevalence),
                                          sample_weight=weights[validation], labels=[0, 1]))
            if validation_loss >= baseline_loss:
                quality_warnings.append("验证损失未优于训练成功率常数基线")
            if stop_reason == "max_epochs":
                quality_warnings.append("达到训练上限，尚未满足早停条件")
        return dict(early_stopping_metric="weighted_log_loss" if self.early_stopping else None,
                    epochs_run=epochs, best_epoch=best_epoch, stop_reason=stop_reason,
                    train_records=len(train), validation_records=len(validation),
                    train_log_loss=float(train_loss), validation_log_loss=validation_loss,
                    validation_baseline_log_loss=baseline_loss,
                    validation_loss_curve=self.validation_loss_curve_,
                    diagnostic_auc=float(roc_auc_score(labels, probability, sample_weight=weights)),
                    probability_min=float(probability.min()), probability_max=float(probability.max()),
                    hidden_layer_sizes=list(self.classifier.hidden_layer_sizes),
                    quality_warnings=quality_warnings)

    def predict_success_probability(self, goals):
        """仅此接口返回成功率；evaluate 返回连续概率密度。"""
        probability = self.classifier.predict_proba(self.scaler.transform(goals))[:, 1]
        return np.maximum(probability, 1e-12)

    def evaluate(self, desired_goals, scale=True, return_density=True):
        goals = np.asarray(desired_goals, dtype=float)
        transformed = self.scaler.transform(goals) if scale else goals
        values = self.density.score_samples(transformed)
        return transformed, np.exp(values) if return_density else values

    def sample(self, n_samples, random_state=0):
        """先按预测权重选择核中心，再从对应高斯核采样。"""
        return self.scaler.inverse_transform(self.density.sample(n_samples, random_state))

    def kl_divergence_uniform_to_kde_integrate(self, samples, dV, u_density):
        return uniform_grid_kl(self.log_density(samples), dV, u_density)
