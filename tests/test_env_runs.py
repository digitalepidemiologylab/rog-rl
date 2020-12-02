import pytest
from rog_rl.env import AgentState
from rog_rl.env import ActionType
from rog_rl.envs.rog_sim_single_agent_env import ActionType as SingleAgentActionType


def test_run_env(env):
    observation = env.reset()
    done = False
    k = 0

    while not done:
        _action = env.action_space.sample()
        print("Action : ", _action)
        observation, reward, done, info = env.step(_action)

def test_run_single_agent_env(single_agent_env):
    observation = single_agent_env.reset()
    done = False
    k = 0

    while not done:
        _action = single_agent_env.action_space.sample()
        print("Action : ", _action)
        observation, reward, done, info = single_agent_env.step(_action)


def test_actions_env(env):
    observation = env.reset()
    _action = env.action_space.sample()
    n_action_types = len(ActionType)
    for i in range(n_action_types):
        _action[0] = i
        print("Action : ", _action)
        observation, reward, done, info = env.step(_action)


def test_actions_single_agent_env(single_agent_env):
    observation = single_agent_env.reset()
    _action = single_agent_env.action_space.sample()
    n_action_types = len(ActionType)
    for i in range(n_action_types):
        _action = i
        print("Action : ", _action)
        observation, reward, done, info = single_agent_env.step(_action)




if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", __file__]))