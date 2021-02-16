from copy import deepcopy, copy

import gym
from gym import wrappers
from gym import Wrapper
import numpy as np
from rog_rl import FixedOrderExplorationEnv  # noqa


class RogRLStateEnv(Wrapper):

    def __init__(self, config={}, name="BaseGridRogRLEnv-v0"):
        # super().__init__(config)
        self.env = gym.make(name, config=config)
        if hasattr(self.env, '_model'):
            self.model = self.env._model
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.running_reward = 0

    def reset(self, *args, **kwargs):
        # super().reset()
        self.running_reward = 0
        obs = self.env.reset(*args, **kwargs)
        if hasattr(self.env, '_model'):
            self.model = self.env._model
        return obs

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.running_reward += rew
        # score = self.running_reward if done else 0
        return obs, rew, done, info

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def set_state(self, state):
        self.running_reward = state[1]
        self.env = copy(state[0])
        self.model = deepcopy(state[2])
        self.env._model = self.model
        obs = np.array(list(self.model.observation))
        return {"obs": obs}

    def get_state(self):
        # del self.env._model
        # del self.env.observation_space
        env = copy(self.env)
        model = deepcopy(self.model)
        # self.env._model = self.model
        return env, self.running_reward, model


if __name__ == "__main__":

    np.random.seed(100)
    render = "ansi"  # "ansi"  # change to "human"
    env_config = dict(
        width=4,
        height=4,
        population_density=1.0,
        vaccine_density=1.0,
        initial_infection_fraction=0.04,
        initial_vaccination_fraction=0,
        prob_infection=0.2,
        prob_agent_movement=0.0,
        disease_planner_config={
            "incubation_period_mu": 0,
            "incubation_period_sigma": 0,
            "recovery_period_mu": 20,
            "recovery_period_sigma": 0,
        },
        vaccine_score_weight=0.5,
        max_simulation_timesteps=20 * 20 * 10,
        early_stopping_patience=20,
        use_renderer=render,  # can be "human", "ansi"
        use_np_model=True,
        toric=False,
        dummy_simulation=False,
        debug=True,
        seed=0)
    env = RogRLStateEnv(config=env_config, name="FreeExplorationEnv-v0")
    print("USE RENDERER ?", env.env.use_renderer)
    record = False
    if record:
        # records the the rendering in the `recording` folder
        env = wrappers.Monitor(env, "recording", force=True)
    observation = env.reset()
    done = False
    k = 0
    states = None
    if not record:
        env.render(mode=render)
    while not done:
        env.action_space.seed(k)
        _action = env.action_space.sample()
        print("Action : ", _action)
        observation, reward, done, info = env.step(_action)
        print(observation.shape)

        if not record:
            env.render(mode=render)
        k += 1
        print("="*100)
        if k == 3:
            # save state
            states = env.get_state()
        if k == 6:
            # reset to saved state
            env.set_state(states)
        # print(observation.shape)
        # print(k, reward, done)
    print(np.sum(observation, axis=0))