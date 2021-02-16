from gym import spaces, wrappers

from enum import Enum
import numpy as np

from rog_rl.agent_state import AgentState
from rog_rl.env import RogRLEnv


class ActionType(Enum):
    STEP = 0
    VACCINATE = 1


class BaseGridRogRLEnv(RogRLEnv):

    def set_observation_space(self):
        return spaces.Box(
            low=np.float32(0),
            high=np.float32(1),
            shape=(
                self.width,
                self.height,
                len(AgentState)))

    def set_action_space(self):
        return spaces.MultiDiscrete(
            [
                len(ActionType), self.width, self.height
            ])

    def set_action_type(self):
        self.action_type = ActionType

    def step_action(self, action):

        _observation = False

        response = "STEP"

        action = [int(x) for x in action]
        if self.debug:
            print("Action : ", action)

        # Handle action propagation in real simulator
        action_type = action[0]
        cell_x = action[1]
        cell_y = action[2]

        response = "STEP"

        if action_type == ActionType.STEP.value:
            self._model.tick()
        elif action_type == ActionType.VACCINATE.value:
            vaccination_success, response = \
                self._model.vaccinate_cell(cell_x, cell_y)

        _observation = self.get_observation()
        return _observation, response


if __name__ == "__main__":
    np.random.seed(0)
    render = "ansi"  # "PIL" # "ansi"  # change to "human"
    env_config = dict(
        width=8,
        height=7,
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
        use_np_model=True,
        max_simulation_timesteps=200,
        early_stopping_patience=20,
        use_renderer=render,
        toric=False,
        dummy_simulation=False,
        debug=True,
        seed=0)
    env = BaseGridRogRLEnv(config=env_config)
    print("USE RENDERER ?", env.use_renderer)
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
        # _action = input("Enter action - ex: [1, 4, 2] : ")
        # if _action.strip() == "":
        #     _action = env.action_space.sample()
        # else:
        #     _action = [int(x) for x in _action.split()]
        #     assert _action[0] in [0, 1]
        #     assert _action[1] in list(range(env._model.width))
        #     assert _action[2] in list(range(env._model.height))
        print("Action : ", _action, "     Step:", k)
        observation, reward, done, info = env.step(_action)
        if not record:
            env.render(mode=render)
        k += 1
        # print(observation.shape)
        # print(k, reward, done)
    print(np.sum(observation, axis=0))
    print(info)
