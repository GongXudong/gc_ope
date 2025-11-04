from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig


def load_config(config_path, config_name) -> DictConfig:
    # 清除之前的 Hydra 实例（避免重复初始化报错）
    GlobalHydra.instance().clear()
    
    # 初始化 Hydra，指定配置文件目录（相对于当前脚本的路径）
    with initialize(version_base=None, config_path=config_path):
        # 组合配置：
        # - config_name 是主配置文件名（不含 .yaml）
        # - overrides 用于覆盖配置（可选，格式：["key=value", ...]）
        cfg = compose(
            config_name=config_name,  # 加载 conf/config.yaml
            # overrides=["db=postgres", "batch_size=64"]  # 覆盖默认配置：使用 postgres，batch_size=64
        )
        return cfg
