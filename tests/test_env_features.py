from rog_rl.env import ActionType
import pytest
import numpy as np
seed = 1


@pytest.mark.skip(reason="TODO: condition that ensures env \
    ends due to zero susceptible")
def test_no_susceptible(all_envs):
    for env in all_envs:
        np.random.seed(seed)
        env.config['early_stopping_patience'] = 10000
        observation = env.reset()
        done = False
        max_steps = env._model.max_timesteps
        step = 0
        susceptible_population = 0
        while not done:
            _action = env.action_space.sample()
            observation, reward, done, info = env.step(_action)
            susceptible_population = info['population.SUSCEPTIBLE']
            step += 1

        if susceptible_population != 0:
            assert max_steps == step


@pytest.mark.skip(reason="TODO: condition that ensures env \
    ends due to early stop")
def test_early_stopping(env):
    np.random.seed(100)
    early_stop = 2
    env.config['early_stopping_patience'] = early_stop
    observation = env.reset()
    done = False
    max_steps = env._model.max_timesteps
    step = 0
    susceptible_population = 0
    previous_susceptible_population = 0
    early_stop_counter = 0
    while not done:
        _action = env.action_space.sample()
        _action[0] = ActionType.STEP.value
        observation, reward, done, info = env.step(_action)
        susceptible_population = info['population.SUSCEPTIBLE']

        if previous_susceptible_population == susceptible_population:
            early_stop_counter += 1
        else:
            early_stop_counter = 0
        previous_susceptible_population = int(susceptible_population)
        step += 1

    print(early_stop, early_stop_counter)
    if early_stop != early_stop_counter:
        assert max_steps == step


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", __file__]))
