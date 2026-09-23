"""流模型共用的加噪训练、按目标分组验证和全数据重拟合流程。"""

import numpy as np
from gc_ope.evaluate.evaluator_common import positive_samples_and_weights


def spatial_validation_split(goals, fraction, random_state):
    """同一个目标坐标的所有副本留在同一侧，避免固定网格的重复点泄漏。

    相比按评估身份分组，此处更严格：不同时间的同一目标也不跨侧。
    只划分本模型已有的成功目标，不接触另一侧估计器的数据。
    """
    unique, inverse = np.unique(goals, axis=0, return_inverse=True)
    if len(unique) < 10:
        return None
    count = max(2, int(np.ceil(len(unique) * fraction)))
    chosen = np.random.default_rng(random_state).permutation(len(unique))[:count]
    valid = np.isin(inverse, chosen)
    return np.flatnonzero(~valid), np.flatnonzero(valid)


class FlowTraining:
    """共享训练流程；两类流只定义模型、训练目标和验证密度。"""

    def _configure_regularization(self, noise_std, early_stopping, validation_fraction,
                                 min_epochs, patience, validation_interval, tol, fallback_epochs):
        if not np.isfinite(noise_std) or noise_std < 0 or not 0 < validation_fraction < .5:
            raise ValueError("训练噪声必须非负有限，验证比例必须在 (0,0.5) 内")
        for value in [min_epochs, patience, validation_interval, fallback_epochs]:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError("轮数、耐心和验证间隔必须为正整数")
        if not np.isfinite(tol) or tol < 0:
            raise ValueError("验证改善阈值必须非负有限")
        self.noise_std, self.early_stopping = float(noise_std), bool(early_stopping)
        self.validation_fraction, self.min_epochs = validation_fraction, min_epochs
        self.patience, self.validation_interval, self.tol = patience, validation_interval, tol
        self.fallback_epochs = fallback_epochs

    def _fit_phase(self, scaled, weights, budget, validation=None):
        """每个阶段重新初始化；随机流局部化，在线拟合不改变 SAC 的随机数。"""
        import torch
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.random_state)
            model = self._new_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        generator = torch.Generator(device="cpu").manual_seed(self.random_state + 17)
        x = torch.as_tensor(scaled, dtype=torch.float32)
        w = torch.as_tensor(weights / weights.sum(), dtype=torch.float32)
        losses, checks = [], []
        best_epoch, best_score, progress_score, last_progress = budget, np.inf, np.inf, 0
        model.train()
        for epoch in range(1, budget + 1):
            loss = self._training_loss(model, x, w, generator)
            if not torch.isfinite(loss):
                raise FloatingPointError("正则化流的训练损失非有限")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            if validation is None or epoch < min(self.min_epochs, budget):
                continue
            if epoch % self.validation_interval != 0 and epoch != budget:
                continue
            vx, vw = validation
            model.eval()
            values = self._validation_log_density(model, vx)
            if not np.isfinite(values).all():
                raise FloatingPointError("正则化流的验证对数密度非有限")
            score = -float(np.average(values, weights=vw))
            checks.append(dict(epoch=epoch, nll=score))
            if score < best_score:
                best_epoch, best_score = epoch, score
            if score < progress_score - self.tol:
                progress_score, last_progress = score, epoch
            model.train()
            if epoch - last_progress >= self.patience:
                break
        model.eval()
        return model, losses, checks, best_epoch

    def fit_evaluator(self):
        """先用留出记录选训练预算，再重新使用所有成功记录进行最终拟合。"""
        import torch
        self._fitted = False
        positive, weights = positive_samples_and_weights(self.eval_res_container)
        positive = self._validate_goals(positive, "positive samples")
        weights = self._validate_weights(weights, len(positive))
        if len(positive) < 2:
            raise ValueError("流模型至少需要两个成功样本")
        torch.set_num_threads(1)
        split = spatial_validation_split(positive, self.validation_fraction, self.random_state + 701)
        checks, selection_loss, train_count, valid_count = [], [], 0, 0
        budget, reason = self.n_epochs, "fixed_budget"
        try:
            if self.early_stopping and split is not None:
                ti, vi = split
                train_count, valid_count = len(ti), len(vi)
                train_scaled = self.scaler.fit_transform(positive[ti]).astype(np.float32)
                valid_scaled = self.scaler.transform(positive[vi]).astype(np.float32)
                _, selection_loss, checks, budget = self._fit_phase(
                    train_scaled, weights[ti], self.n_epochs, (valid_scaled, weights[vi]))
                reason = "validation_selected"
            elif self.early_stopping:
                # 早期成功坐标太少时，不伪造验证集；显式采用预先配置的短预算。
                budget, reason = min(self.n_epochs, self.fallback_epochs), "insufficient_unique_goals"
            # 只保留选出的轮数，不复用留出阶段的模型参数或 scaler。
            scaled = self.scaler.fit_transform(positive).astype(np.float32)
            model, losses, _, _ = self._fit_phase(scaled, weights, budget)
            self._install_model(model)
            self._fitted = True
            _, log_density = self.evaluate(scaled, scale=False, return_density=False)
            if not np.isfinite(log_density).all():
                raise FloatingPointError("最终拟合密度非有限")
        except Exception:
            self._fitted = False
            raise
        self._loss_curve = losses
        warnings = []
        if checks and budget == self.n_epochs:
            warnings.append("最佳验证轮数位于预算上限，请检查是否仍需训练")
        self.fit_diagnostics_ = dict(
            n_positive_samples=len(positive), n_unique_goals=len(np.unique(positive, axis=0)),
            noise_std=self.noise_std, n_epochs=budget, max_epochs=self.n_epochs,
            early_stopping=self.early_stopping, selection_reason=reason,
            selection_training_records=train_count, selection_validation_records=valid_count,
            validation_group="exact_goal_coordinates", validation_curve=checks,
            validation_space="training-standardized", selection_loss_curve=selection_loss,
            final_refit_records=len(positive), final_loss_first=losses[0], final_loss_last=losses[-1],
            quality_warnings=warnings, training_scheme="noisy targets + weighted validation + full refit",
            random_state=self.random_state, hidden_layer_sizes=list(self.hidden_layer_sizes), device="cpu")
        return positive, scaled, weights, np.exp(np.clip(log_density, -745, 709))


