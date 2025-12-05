import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
from scipy.integrate import nquad
from scipy.special import logsumexp

class UniformDistribution3D:
    """三维连续空间G上的均匀分布"""
    def __init__(self, bounds):
        """
        bounds: list of tuples [(min1, max1), (min2, max2), (min3, max3)]
        定义三维空间的边界
        """
        self.bounds = np.array(bounds)
        self.dims = len(bounds)
        self.volume = np.prod(self.bounds[:, 1] - self.bounds[:, 0])
        self.density = 1.0 / self.volume  # 均匀分布的常数概率密度
        
    def pdf(self, X):
        """计算均匀分布的概率密度函数"""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        # 检查点是否在边界内
        in_bounds = np.all((X >= self.bounds[:, 0]) & (X <= self.bounds[:, 1]), axis=1)
        pdf_values = np.zeros(len(X))
        pdf_values[in_bounds] = self.density
        return pdf_values
    
    def logpdf(self, X):
        """计算均匀分布的对数概率密度"""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        in_bounds = np.all((X >= self.bounds[:, 0]) & (X <= self.bounds[:, 1]), axis=1)
        logpdf_values = np.full(len(X), -np.inf)
        logpdf_values[in_bounds] = np.log(self.density)
        return logpdf_values
    
    def sample(self, n_samples=1000):
        """从均匀分布中采样"""
        samples = np.zeros((n_samples, self.dims))
        for i in range(self.dims):
            samples[:, i] = np.random.uniform(
                self.bounds[i, 0], 
                self.bounds[i, 1], 
                n_samples
            )
        return samples

def estimate_kl_divergence_mc(u, p_kde, n_samples=10000):
    """
    使用蒙特卡洛方法估计KL(u||p)
    KL(u||p) = E_{x~u}[log u(x) - log p(x)]
    """
    # 从均匀分布u中采样
    samples = u.sample(n_samples)
    
    # 计算u的对数概率密度
    log_u = u.logpdf(samples)
    
    # 计算KDE的对数概率密度
    log_p = p_kde.score_samples(samples)
    
    # 计算KL散度
    kl_divergence = np.mean(log_u - log_p)
    
    return kl_divergence

def estimate_kl_divergence_adaptive(u, p_kde, n_iterations=5, initial_samples=1000):
    """
    自适应蒙特卡洛方法，根据p的分布调整采样
    """
    total_kl = 0
    total_weight = 0
    
    for i in range(n_iterations):
        # 从均匀分布中采样
        n_samples = initial_samples * (2**i)
        samples = u.sample(n_samples)
        
        # 计算对数概率密度
        log_u = u.logpdf(samples)
        log_p = p_kde.score_samples(samples)
        
        # 计算重要性权重（避免极端值）
        weights = np.ones_like(log_u)
        mask = np.isfinite(log_p) & np.isfinite(log_u)
        
        if np.sum(mask) > 0:
            kl_batch = np.mean((log_u - log_p)[mask])
            # 使用指数移动平均
            weight = 1.0 / (2**i)
            total_kl = total_kl * (1 - weight) + kl_batch * weight
            total_weight = total_weight * (1 - weight) + weight
    
    return total_kl

def create_test_data(n_samples=500, bounds=[(0, 1), (0, 1), (0, 1)]):
    """
    创建模拟测试数据
    假设policy能够完成的目标集中在某个区域
    """
    # 创建一些测试点，集中在空间的一个区域
    bounds = np.array(bounds)
    center = np.mean(bounds, axis=1)
    
    # 生成集中在中心附近的数据（模拟policy能够完成的目标）
    test_points = []
    for _ in range(n_samples):
        # 从多元正态分布采样，集中在空间中心
        point = np.random.normal(loc=center, scale=0.2, size=3)
        # 确保点在边界内
        point = np.clip(point, bounds[:, 0], bounds[:, 1])
        test_points.append(point)
    
    return np.array(test_points)

def main():
    # 1. 定义三维目标空间G和均匀分布u
    # 假设G是[0,1]×[0,1]×[0,1]的三维空间
    # bounds = [(0, 1), (0, 1), (0, 1)]
    bounds = [(150, 250), (-10, 10), (-30, 30)]
    u = UniformDistribution3D(bounds)
    
    print(f"均匀分布边界: {bounds}")
    print(f"均匀分布体积: {u.volume}")
    print(f"均匀分布密度: {u.density:.6f}")
    
    # 2. 创建测试数据并拟合KDE
    print("\n生成测试数据并拟合KDE...")
    test_data = create_test_data(n_samples=500, bounds=bounds)
    print(f"测试数据形状: {test_data.shape}")
    
    # 使用KernelDensity拟合分布p
    # 通过交叉验证选择带宽
    from sklearn.model_selection import GridSearchCV
    
    # 定义带宽搜索范围
    bandwidths = np.logspace(-2, 0, 10)
    
    # 使用GridSearchCV选择最佳带宽
    grid = GridSearchCV(
        KernelDensity(kernel='gaussian'),
        {'bandwidth': bandwidths},
        cv=5
    )
    grid.fit(test_data)
    
    # 使用最佳带宽创建KDE
    best_bandwidth = grid.best_params_['bandwidth']
    print(f"最佳带宽: {best_bandwidth:.4f}")
    
    kde = KernelDensity(bandwidth=best_bandwidth, kernel='gaussian')
    kde.fit(test_data)
    
    # 3. 计算KL散度
    print("\n计算KL散度...")
    
    # 方法1: 标准蒙特卡洛方法
    kl_mc = estimate_kl_divergence_mc(u, kde, n_samples=10000)
    print(f"KL(u||p) - 蒙特卡洛方法: {kl_mc:.4f}")
    
    # 方法2: 自适应蒙特卡洛方法
    kl_adaptive = estimate_kl_divergence_adaptive(u, kde, n_iterations=5, initial_samples=1000)
    print(f"KL(u||p) - 自适应方法: {kl_adaptive:.4f}")
    
    # 4. 可视化结果（可选）
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(15, 5))
        
        # 子图1: 测试数据分布
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(test_data[:, 0], test_data[:, 1], test_data[:, 2], 
                   alpha=0.5, s=10)
        ax1.set_title('测试数据分布')
        ax1.set_xlabel('维度1')
        ax1.set_ylabel('维度2')
        ax1.set_zlabel('维度3')
        
        # 子图2: 从均匀分布采样
        ax2 = fig.add_subplot(132, projection='3d')
        uniform_samples = u.sample(200)
        ax2.scatter(uniform_samples[:, 0], uniform_samples[:, 1], 
                   uniform_samples[:, 2], alpha=0.5, s=10, color='green')
        ax2.set_title('均匀分布采样')
        ax2.set_xlabel('维度1')
        ax2.set_ylabel('维度2')
        ax2.set_zlabel('维度3')
        
        # 子图3: 从KDE采样
        ax3 = fig.add_subplot(133, projection='3d')
        kde_samples = kde.sample(200)
        ax3.scatter(kde_samples[:, 0], kde_samples[:, 1], 
                   kde_samples[:, 2], alpha=0.5, s=10, color='red')
        ax3.set_title('KDE分布采样')
        ax3.set_xlabel('维度1')
        ax3.set_ylabel('维度2')
        ax3.set_zlabel('维度3')
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib未安装，跳过可视化")
    
    return u, kde, kl_mc

if __name__ == "__main__":
    u, kde, kl_divergence = main()
    
    # 额外功能：计算JS散度（对称版本）
    def estimate_js_divergence(u, kde, n_samples=10000):
        """计算Jensen-Shannon散度"""
        samples_u = u.sample(n_samples)
        samples_p = kde.sample(n_samples)
        
        # 混合分布
        samples_m = np.vstack([samples_u, samples_p])
        
        # 计算对数概率
        log_u_u = u.logpdf(samples_u)
        log_p_u = kde.score_samples(samples_u)
        
        log_u_p = u.logpdf(samples_p)
        log_p_p = kde.score_samples(samples_p)
        
        log_m_u = np.log(0.5 * (np.exp(log_u_u) + np.exp(log_p_u)))
        log_m_p = np.log(0.5 * (np.exp(log_u_p) + np.exp(log_p_p)))
        
        # KL散度
        kl_u_m = np.mean(log_u_u - log_m_u)
        kl_p_m = np.mean(log_p_p - log_m_p)
        
        # JS散度
        js_divergence = 0.5 * (kl_u_m + kl_p_m)
        
        return js_divergence
    
    js_div = estimate_js_divergence(u, kde)
    print(f"\nJS散度（对称度量）: {js_div:.4f}")