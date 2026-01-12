from stable_baselines3 import PPO, SAC
from stable_baselines3.common.distributions import DiagGaussianDistribution
import gymnasium as gym
import torch as th

# ========== 离散动作（PPO）：CartPole-v1（逻辑不变，保持兼容） ==========
env_discrete = gym.make("CartPole-v1")
model_ppo = PPO("MlpPolicy", env_discrete, verbose=0)

# 重置环境获取初始观测
obs_discrete, _ = env_discrete.reset()
# 转换为张量（添加batch维度，指定类型和设备）
obs_tensor_ppo = th.tensor(obs_discrete[None, :], dtype=th.float32).to(model_ppo.device)

# 获取PPO动作分布、采样动作、计算pscore
distribution_ppo = model_ppo.policy.get_distribution(obs_tensor_ppo)
action_ppo = distribution_ppo.sample()
action_log_prob_ppo = distribution_ppo.log_prob(action_ppo)
pscore_ppo = action_log_prob_ppo.exp().item()

print(f"【离散动作】采样动作: {action_ppo.item()}, 对应的pscore（概率）: {pscore_ppo:.4f}")
env_discrete.close()

# ========== 连续动作（SAC）：Pendulum-v1（修复核心逻辑） ==========
env_continuous = gym.make("Pendulum-v1")
model_sac = SAC("MlpPolicy", env_continuous, verbose=0)

# 重置环境获取初始观测
obs_continuous, _ = env_continuous.reset()
# 转换为张量（添加batch维度，指定类型和设备）
obs_tensor_sac = th.tensor(obs_continuous[None, :], dtype=th.float32).to(model_sac.device)

# Step1：通过SAC的Actor获取动作分布的参数
model_sac.policy.actor(obs_tensor_sac) # 前向传播以更新动作分布参数
action_dist = model_sac.policy.actor.action_dist
# Step2：采样动作（自动裁剪到环境动作空间范围）
action_sac = action_dist.sample()
# Step3：计算该动作的pscore（概率密度值）
action_log_prob_sac = action_dist.log_prob(action_sac)
pscore_sac = action_log_prob_sac.exp().item()

# 输出结果（转换为CPU numpy数组，方便展示）
print(f"【连续动作】采样动作: {action_sac.cpu().detach().numpy().flatten()}, 对应的pscore（密度）: {pscore_sac:.6f}")
env_continuous.close()