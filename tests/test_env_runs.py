from rog_rl.env import ActionType
import pytest
import numpy as np

seed = 1000  # 10000 fails


def test_run_all_envs(all_envs):
    for env in all_envs:
        np.random.seed(seed)
        observation = env.reset()
        done = False

        while not done:
            _action = env.action_space.sample()
            observation, reward, done, info = env.step(_action)


def test_actions_env(env):
    np.random.seed(seed)
    observation = env.reset()
    _action = env.action_space.sample()
    n_action_types = len(ActionType)
    for i in range(n_action_types):
        _action[0] = i
        print("Action : ", _action)
        observation, reward, done, info = env.step(_action)


def test_actions_free_exploration_env(free_exploration_env):
    np.random.seed(seed)
    observation = free_exploration_env.reset()
    _action = free_exploration_env.action_space.sample()
    n_action_types = len(ActionType)
    for i in range(n_action_types):
        _action = i
        print("Action : ", _action)
        observation, reward, done, info = free_exploration_env.step(_action)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
