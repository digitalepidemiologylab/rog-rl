from rog_rl import model_np
from rog_rl import RogSimEnv  # noqa
from rog_rl import RogSimSingleAgentEnv  # noqa
from rog_rl.env import ActionType

from rog_rl.envs.rog_sim_single_agent_env import ActionType as SingleAgentActionType

from rog_rl.agent_state import AgentState
from gym import wrappers
import pytest
import gym
import numpy as np
import time

"""Tests for `rogi_rl` gym based env."""

names = ['RogRL-v0','RogRLSingleAgent-v0']


@pytest.mark.parametrize('name, width, height, toric, dummy_simulation, \
                         use_renderer,use_np_model', [
    # (names[0], 5, 5, False, False, "human", True),
    (names[0], 10, 10, True, False, "ansi", True),
    (names[0], 10, 10, True, False, "PIL", True),
    (names[0], 10, 10, True, False, "simple", True),
    (names[0], 10, 20, True, False, "simple", False),
    (names[0], 20, 20, True, False, False, True),
    (names[0], 20, 20, False, False, False, True),
    # (names[1], 5, 5, False, False, "human", True),
    (names[1], 10, 10, True, False, "ansi", True),
    (names[1], 10, 10, True, False, "PIL", True),
    (names[1], 10, 10, True, False, "simple", True),
    (names[1], 10, 20, True, False, "simple", False),
    (names[1], 20, 20, True, False, False, True),
    (names[1], 20, 20, False, False, False, True),
])
def test_env_instantiation(name, width, height, toric, dummy_simulation,
                           use_renderer,use_np_model):
    """
    Test that standard gym env actions
    methods like reset, step
    """
    seed = 1
    n_states = len(AgentState)
    n_action_types = len(ActionType)

    # Renderer is disabled for now but it can be enabled
    # To run different renders, comment below line
    # use_renderer = False

    env_config = dict(
        width=width,
        height=height,
        toric=toric,
        dummy_simulation=dummy_simulation,
        use_renderer=use_renderer,
        debug=True,
        np_random=seed,
        use_np_model=use_np_model
    )
    env = gym.make(name, config=env_config)
    np.random.seed(seed)

    if use_renderer:
        env = wrappers.Monitor(env,"recording",force=True)
    observation = env.reset()
    assert observation.shape[:2] == (width, height)
    for i in range(n_action_types):
        action = env.action_space.sample()
        # Ensure we do both step and vaccinate
        state, reward, done, info = env.step(action)

        assert isinstance(state, np.ndarray)
        assert state.shape[:2] == (width, height)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        if use_renderer:
            if use_renderer in ["human","PIL"]:
                use_renderer = "rgb_array"
            frame = env.render(use_renderer)
            assert frame is not None
            if use_renderer != "ansi":
                if not isinstance(frame, (np.ndarray, np.generic)):
                    raise Exception('Wrong type {} for {} (must be np.ndarray or np.generic)'.format(type(frame), frame))
                if frame.dtype != np.uint8:
                    raise Exception("Your frame has data type {}, but we require uint8 (i.e. RGB values from 0-255).".format(frame.dtype))


    env.close()


def test_make():
    for name in names:
        env = gym.make(name)
        assert env.spec.id == name
        assert isinstance(env, gym.Env)


def test_monitor():
    for name in names:
        env = gym.make(name)
        env = wrappers.Monitor(env, "recording", force=True)
        assert env.spec.id == name
        assert isinstance(env, wrappers.Monitor)


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", __file__]))