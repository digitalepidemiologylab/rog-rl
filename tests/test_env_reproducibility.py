import pytest
import numpy as np
import random


n_runs = 1
seed = 1
np.random.seed(seed)
random.seed(seed)


def test_env_reproducable(all_envs):
    # all_envs.extend(all_mesa_envs)
    for env in all_envs:
        env.seed(1)
        observation = env.reset()
        done = False
        k = 0
        while not done:
            env.action_space.seed(k)
            _action = env.action_space.sample()
            observation, reward, done, info = env.step(_action)
            k += 1
        final_obs = np.sum(observation, axis = 0)

        env.seed(1)
        observation = env.reset()
        done = False
        k = 0
        while not done:
            env.action_space.seed(k)
            _action = env.action_space.sample()
            observation, reward, done, info = env.step(_action)
            k += 1

        assert np.sum(final_obs - np.sum(observation, axis = 0)) == 0


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", __file__]))
