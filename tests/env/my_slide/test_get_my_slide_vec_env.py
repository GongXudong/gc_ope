# from gc_ope.env.get_vec_env import get_vec_env
from gc_ope.env.get_vec_env import get_vec_env
from gc_ope.utils.load_config_with_hydra import load_config


def test_get_myslide_vec_env():
    cfg = load_config(
        config_path="../../../configs/train",
        config_name="config",
    )

    cfg.env.env_id = "MySlideSparse-v0"
    cfg.env.train_env.num_process = 2
    cfg.env.evaluation_env.num_process = 1
    cfg.env.callback_env.num_process = 1

    train_env, eval_env, callback_env = get_vec_env(cfg.env)

    # check env
    obs = train_env.reset()

    for i in range(10):
        actions = [train_env.action_space.sample() for _ in range(train_env.num_envs)]
        # action = train_env.action_space.sample()
        # print(action.shape)
        obs, reward, done, info = train_env.step(actions)
        print(actions, obs, reward)

    print(train_env.get_attr(attr_name="unwrapped", indices=[0])[0].task.distance_threshold)
    # print(train_env.get_attr(attr_name="unwrapped.task.distance_threshold", indices=[0]))

if __name__ == "__main__":
    test_get_myslide_vec_env()
    