from gym import spaces, wrappers

from enum import Enum
import numpy as np


from rog_rl.vaccination_response import VaccinationResponse
from rog_rl import FreeExplorationEnv


class ActionType(Enum):
    MOVE = 0
    VACCINATE = 1


class FixedOrderExplorationEnv(FreeExplorationEnv):
    """
    A single agent env contains a single
    vaccination agent which can move around the grid
    and apart from choose to vaccinate any cell
    or
    step ahead in the internal disease simulator.
    """

    def __init__(self, config={}):
        # Setup Config
        super().__init__(config)
        self.agent_reset()

    def env_reset(self):
        # Initialize location of vaccination agent
        self.agent_reset()

        self._game_steps = 1

    def agent_reset(self):
        self.vacc_agent_x = 0
        self.vacc_agent_y = 0

    def set_action_space(self):
        """
        The action space is composed of 2 discrete actions :

        MOVE : Moves the vaccination-agent in fixed order

        VACCINATE : Vaccinates the current location of the vaccination-agent
        """
        return spaces.Discrete(len(ActionType))

    def set_action_type(self):
        self.action_type = ActionType

    def move_action(self):
        if self.vacc_agent_x == self.width - 1:
            if self.vacc_agent_y == self.height - 1:
                # Navigation Complete - Move to next time step
                self.step_tick()
                self.agent_reset()
            else:
                self.vacc_agent_y += 1
                self.vacc_agent_y %= self.height
                self.vacc_agent_x += 1
                self.vacc_agent_x %= self.width
        else:
            self.vacc_agent_x += 1
            self.vacc_agent_x %= self.width

    def step_action(self, action):

        _observation = False

        response = "MOVE"

        if action == ActionType.MOVE.value:
            """
            Handle moving to next cell action
            """
            _observation = self._model.get_observation()
            self.move_action()

        elif action == ActionType.VACCINATE.value:
            """
            Handle VACCINATE action
            """
            # Vaccinate the cell where the vaccination agent currently is
            cell_x = self.vacc_agent_x
            cell_y = self.vacc_agent_y
            vaccination_success, response = self._model.vaccinate_cell(cell_x, cell_y)
            _observation = self._model.get_observation()

            # Force Run simulation to completion if
            # run out of vaccines
            if response == VaccinationResponse.AGENT_VACCINES_EXHAUSTED:
                _observation, _ = self.step_tick()

            self.move_action()

        return _observation, response


if __name__ == "__main__":

    np.random.seed(100)
    render = "ansi"  # "ansi"  # change to "human"
    env_config = dict(
        width=5,
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
        vaccine_score_weight=0.5,
        max_simulation_timesteps=20 * 20 * 10,
        early_stopping_patience=20,
        use_renderer=render,  # can be "human", "ansi"
        use_np_model=True,
        toric=False,
        dummy_simulation=False,
        simulation_single_tick=True,
        debug=True,
        seed=0,
    )
    env = FixedOrderExplorationEnv(config=env_config)
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
        print("Action : ", _action, "     Step:", k)
        observation, reward, done, info = env.step(_action)
        print(observation.shape)

        if not record:
            env.render(mode=render)
        print("Vacc_agent_location : ", env.vacc_agent_x, env.vacc_agent_y)
        k += 1
        print("=" * 100)
        # print(observation.shape)
        # print(k, reward, done)
    print(np.sum(observation, axis=0))
    print(info)
