import numpy as np
from sklearn.neighbors import KernelDensity


def kl_divergence_kde_3d_monte_carlo(kde_p: KernelDensity, kde_q: KernelDensity, n_samples: int=100000) -> float:
    """
    使用蒙特卡洛方法计算三维KDE的KL散度
    更适合高维情况
    """
    # 从分布P中采样
    samples_p = kde_p.sample(n_samples)
    
    # 计算这些样本在两个分布下的对数概率密度
    log_p = kde_p.score_samples(samples_p)
    log_q = kde_q.score_samples(samples_p)
    
    # 避免数值问题
    mask = np.exp(log_q) > 1e-10
    log_p_masked = log_p[mask]
    log_q_masked = log_q[mask]
    
    # KL散度估计
    kl_value = np.mean(log_p_masked - log_q_masked)
    
    return kl_value

def kl_divergence_kde_3d(kde_p, kde_q, bounds=None, sample_points=50):
    """
    计算两个三维KDE分布之间的KL散度
    
    参数:
    kde_p, kde_q: 训练好的三维KernelDensity对象
    bounds: 积分边界 [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
    sample_points: 每个维度的采样点数
    """
    
    # 设置默认积分边界
    if bounds is None:
        bounds = [[-5, 5], [-5, 5], [-5, 5]]
    
    # 创建三维网格
    x = np.linspace(bounds[0][0], bounds[0][1], sample_points)
    y = np.linspace(bounds[1][0], bounds[1][1], sample_points)
    z = np.linspace(bounds[2][0], bounds[2][1], sample_points)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    
    # 计算概率密度
    log_p = kde_p.score_samples(grid_points)
    log_q = kde_q.score_samples(grid_points)
    
    p = np.exp(log_p)
    
    # 避免数值问题
    mask = (p > 1e-10) & (np.exp(log_q) > 1e-10)
    # grid_points_masked = grid_points[mask]
    p_masked = p[mask]
    log_p_masked = log_p[mask]
    log_q_masked = log_q[mask]
    
    # 重新组织数据用于积分
    # points_3d = grid_points_masked.reshape(-1, 3)
    # x_masked = points_3d[:, 0]
    # y_masked = points_3d[:, 1]
    # z_masked = points_3d[:, 2]
    
    # 计算KL散度 - 使用三重积分
    integrand = p_masked * (log_p_masked - log_q_masked)
    
    # 由于网格不规则，使用近似积分
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    dV = dx * dy * dz
    
    kl_value = np.sum(integrand) * dV
    
    return kl_value