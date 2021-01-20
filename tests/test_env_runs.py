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


def test_actions_single_agent_env(single_agent_env):
    np.random.seed(seed)
    observation = single_agent_env.reset()
    _action = single_agent_env.action_space.sample()
    n_action_types = len(ActionType)
    for i in range(n_action_types):
        _action = i
        print("Action : ", _action)
        observation, reward, done, info = single_agent_env.step(_action)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", __file__]))
