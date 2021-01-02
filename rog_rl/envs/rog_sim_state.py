from copy import deepcopy

import gym
from gym import spaces, wrappers
from gym import Wrapper
import numpy as np
from gym.spaces import Discrete, Dict, Box
from rog_rl import RogSimSingleAgentActionEnv


class RogSimState(Wrapper):
    """
    Wrapper for gym CartPole environment where the reward
    is accumulated to the end
    """

    def __init__(self, config={}):
        # super().__init__(config)
        self.env = gym.make("RogRLSingleAgentAction-v0", config = config)
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
        return {"obs": obs}

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.running_reward += rew
        score = self.running_reward if done else 0
        return {"obs": obs}, score, done, info

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def set_state(self, state):
        self.running_reward = state[1]
        self.env = deepcopy(state[0])
        self.model = deepcopy(state[0])
        obs = np.array(list(self.env.unwrapped.state))
        return {"obs": obs}

    def get_state(self):
        return deepcopy(self.env), self.running_reward


if __name__ == "__main__":

    np.random.seed(100)
    render = "ansi" # "ansi"  # change to "human"
    env_config = dict(
                    width=5,
                    height=5,
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
                    seed = 0)
    env = RogSimState(config=env_config)
    print("USE RENDERER ?", env.env.use_renderer)
    record = True
    if record:
        # records the the rendering in the `recording` folder
        env = wrappers.Monitor(env, "recording", force=True)
    observation = env.reset()
    done = False
    k = 0
    if not record:
        env.render(mode=render)
    while not done:
        env.action_space.seed(k)
        _action = env.action_space.sample()
        print("Action : ", _action)
        observation, reward, done, info = env.step(_action)
        print(observation["obs"].shape)

        if not record:
            env.render(mode=render)
        print("Vacc_agent_location : ", env.vacc_agent_x, env.vacc_agent_y)
        k += 1
        print("="*100)
        # print(observation.shape)
        # print(k, reward, done)
    print(np.sum(observation["obs"],axis=0))
