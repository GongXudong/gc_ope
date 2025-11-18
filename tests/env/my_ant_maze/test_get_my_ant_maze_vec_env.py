from gc_ope.env.get_vec_env import get_vec_env
from gc_ope.utils.load_config_with_hydra import load_config
from gc_ope.env.utils.my_maze.register_env import register_my_ant_maze

register_my_ant_maze()

def test_get_antmaze_vec_env():

    

    cfg = load_config(
        config_path="../../../configs/train",
        config_name="config",
    )

    cfg.env = dict(
        env_id = "MyAntMaze_Medium_Diverse_G-v3",
        continuing_task = False,
        reward_type = "sparse",
        maze_map = [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, "r", "g", 1, 1, "g", "g", 1],
            [1, "g", "g", 1, "g", "g", "g", 1],
            [1, 1, "g", "g", "g", 1, 1, 1],
            [1, "g", "g", 1, "g", "g", "g", 1],
            [1, "g", 1, "g", "g", 1, "g", 1],
            [1, "g", "g", "g", 1, "g", "g", 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
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
        env_str = "my_antmaze",
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
    test_get_antmaze_vec_env()
