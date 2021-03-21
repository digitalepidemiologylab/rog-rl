from rog_rl import BaseGridRogRLEnv  # noqa
from rog_rl import FreeExplorationEnv  # noqa
from rog_rl.env import ActionType

from gym import wrappers
import pytest
import gym
import numpy as np

"""Tests for `rogi_rl` gym based env."""

names = [
    "BaseGridRogRLEnv-v0",
    "FreeExplorationEnv-v0",
    "FixedOrderExplorationEnv-v0",
    "RogRLStateEnv-v0",
]


@pytest.mark.parametrize(
    "name, width, height, toric, dummy_simulation, \
                         use_renderer,use_np_model,simulation_single_tick",
    [
        (names[0], 5, 5, False, False, "human", True, True),
        (names[0], 10, 10, True, False, "ansi", True, True),
        (names[0], 10, 10, True, False, "PIL", True, False),
        (names[0], 10, 10, True, False, "simple", True, True),
        # (names[0], 10, 20, True, False, "simple", False, False),
        (names[0], 20, 20, True, False, False, True, False),
        (names[0], 20, 20, False, False, False, True, True),
        (names[1], 5, 5, False, False, "human", True, True),
        (names[1], 10, 10, True, False, "ansi", True, True),
        (names[1], 10, 10, True, False, "PIL", True, False),
        (names[1], 10, 10, True, False, "simple", True, True),
        # (names[1], 10, 20, True, False, "simple", False, False),
        (names[1], 20, 20, True, False, False, True, False),
        (names[1], 20, 20, False, False, False, True, True),
        (names[2], 5, 5, False, False, "human", True, True),
        (names[2], 10, 10, True, False, "ansi", True, True),
        (names[2], 10, 10, True, False, "PIL", True, False),
        (names[2], 10, 10, True, False, "simple", True, True),
        # (names[2], 10, 20, True, False, "simple", False, False),
        (names[2], 20, 20, True, False, False, True, False),
        (names[2], 20, 20, False, False, False, True, True),
        (names[3], 5, 5, False, False, "human", True, True),
        (names[3], 10, 10, True, False, "ansi", True, True),
        (names[3], 10, 10, True, False, "PIL", True, False),
        (names[3], 10, 10, True, False, "simple", True, True),
        # (names[0], 10, 20, True, False, "simple", False, False),
        (names[3], 20, 20, True, False, False, True, False),
        (names[3], 20, 20, False, False, False, True, True),
    ],
)
def test_env_instantiation(
    name,
    width,
    height,
    toric,
    dummy_simulation,
    use_renderer,
    use_np_model,
    simulation_single_tick,
):
    """
    Test that standard gym env actions
    methods like reset, step
    """
    seed = 1
    n_action_types = len(ActionType)
    done = False
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
        use_np_model=use_np_model,
        simulation_single_tick=simulation_single_tick,
    )
    env = gym.make(name, config=env_config)
    np.random.seed(seed)

    if use_renderer:
        env = wrappers.Monitor(env, "recording", force=True)
    observation = env.reset()
    assert observation.shape[:2] == (width, height)
    state, reward, done, info = None, None, False, None
    for i in range(n_action_types):
        action = env.action_space.sample()
        # Ensure we do both step and vaccinate
        if not done:
            state, reward, done, info = env.step(action)

        assert isinstance(state, np.ndarray)
        assert state.shape[:2] == (width, height)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        if use_renderer:
            if use_renderer in ["human", "PIL"]:
                use_renderer = "rgb_array"
            frame = env.render(use_renderer)
            assert frame is not None
            if use_renderer != "ansi":
                if not isinstance(frame, (np.ndarray, np.generic)):
                    raise Exception(
                        "Wrong type {} for {} (must be \
                        np.ndarray or np.generic)".format(
                            type(frame), frame
                        )
                    )
                if frame.dtype != np.uint8:
                    raise Exception(
                        "Your frame has data type {}, but we require \
                            uint8 (i.e. RGB values from \
                                0-255).".format(
                            frame.dtype
                        )
                    )

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
