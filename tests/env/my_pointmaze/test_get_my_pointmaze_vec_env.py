from gc_ope.env.get_vec_env import get_vec_env
from gc_ope.utils.load_config_with_hydra import load_config


def test_get_pointmaze_vec_env():
    cfg = load_config(
        config_path="../../../configs/train",
        config_name="config",
    )

    cfg.env = dict(
        env_id = "MyPointMaze_Large_Diverse_G-v3",
        continuing_task = False,
        maze_map = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, "r", "g", "g", "g", 1, "g", "g", "g", "g", "g", 1],
            [1, "g", 1, 1, "g", 1, "g", 1, "g", 1, "g", 1],
            [1, "g", "g", "g", "g", "g", "g", 1, "g", "g", "g", 1],
            [1, "g", 1, 1, 1, 1, "g", 1, 1, 1, "g", 1],
            [1, "g", "g", 1, "g", 1, "g", "g", "g", "g", "g", 1],
            [1, 1, "g", 1, "g", 1, "g", 1, "g", 1, 1, 1],
            [1, "g", "g", 1, "g", "g", "g", 1, "g", "g", "g", 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ],
        train_env = dict(
            num_process=2,
            seed=3,
        ),
        evaluation_env = dict(
            num_process=2,
            seed=3,
        ),
        callback_env = dict(
            num_process=2,
            seed=3,
        ),
        env_str = "pointmaze",
    )

    train_env, eval_env, callback_env = get_vec_env(cfg.env)

    print(train_env.observation_space, train_env.action_space)
    
    # check env
    obs = train_env.reset()

    print("Initial obs: ", obs)

    for i in range(10):
        actions = [train_env.action_space.sample() for _ in range(train_env.num_envs)]
        # action = train_env.action_space.sample()
        # print(action.shape)
        obs, reward, done, info = train_env.step(actions)
        print(actions, obs, reward)

    print(train_env.get_attr(attr_name="unwrapped", indices=[0])[0])
    # print(train_env.get_attr(attr_name="unwrapped.task.distance_threshold", indices=[0]))

if __name__ == "__main__":
    test_get_pointmaze_vec_env()
