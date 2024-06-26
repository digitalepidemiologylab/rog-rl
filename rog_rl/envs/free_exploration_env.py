from gym import spaces, wrappers

from enum import Enum
import numpy as np


from rog_rl.vaccination_response import VaccinationResponse
from rog_rl.env import RogRLEnv


class ActionType(Enum):
    MOVE_N = 0
    MOVE_E = 1
    MOVE_W = 2
    MOVE_S = 3
    VACCINATE = 4
    SIM_TICK = 5


class FreeExplorationEnv(RogRLEnv):
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

        self.vacc_agent_x = 0
        self.vacc_agent_y = 0

        self._game_steps = 0

    def env_reset(self):
        # Initialize location of vaccination agent
        self.vacc_agent_x = self.np_random.randint(self.width)
        self.vacc_agent_y = self.np_random.randint(self.height)

        self._game_steps = 1

    def update_env_renderer_stats(self):
        # Add VACC_AGENT coords to render state
        self.renderer.update_stats("VACC_AGENT_X", str(self.vacc_agent_x))
        self.renderer.update_stats("VACC_AGENT_Y", str(self.vacc_agent_y))

    def set_observation_space(self):
        # In case we club the Exposed, Symptomatic, Infectious, and Vaccinated
        """
        The observation space in this case will be of shape (width, height, 3)
        where we represent 4 channels of information across the grid

        Channel 1 : Is the cell Susceptible
        Channel 2 : Is an agent in this cell infecting neighbours
        Channel 3 : Is an agent in this cell recovered
        Channel 4 : Is an agent in this cell Vaccinated
        Channel 5 : Is the vaccination agent here
        """
        self.observation_channels = 5
        return spaces.Box(
            low=np.uint8(0),
            high=np.uint8(1),
            shape=(self.width, self.height, self.observation_channels),
        )

    def set_action_space(self):
        """
        The action space is composed of 5 discrete actions :

        MOVE_N : Moves the vaccination-agent north
        MOVE_E : Moves the vaccination-agent east
        MOVE_W : Moves the vaccination-agent west
        MOVE_S : Moves the vaccination-agent south

        VACCINATE : Vaccinates the current location of the vaccination-agent
        SIM_TICK : adds a simulation tick to the disease model
        """
        return spaces.Discrete(len(ActionType))

    def set_action_type(self):
        self.action_type = ActionType

    def post_process_observation(self, observation):
        """
        Channel 1 : Is the cell Susceptible
        Channel 2 : Is an agent in this cell exposed/infected at some point
        Channel 3 : Is an agent in this cell Vaccinated
        Channel 4 : Is the vaccination agent here

        """
        vaccination_agent_channel = np.zeros((self.width, self.height))
        vaccination_agent_channel[self.vacc_agent_x, self.vacc_agent_y] = 1

        p_obs = np.dstack([observation, vaccination_agent_channel])
        return np.uint8(p_obs)

    def step_action(self, action):

        _observation = False

        response = "MOVE"

        if action == ActionType.SIM_TICK.value:
            """
            Handle SIM_TICK action
            """
            # Handle action propagation in real simulator
            _observation, response = self.step_tick()

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

        elif action == ActionType.MOVE_N.value:
            """
            Handle MOVE_N action
            """
            self.vacc_agent_y -= 1
            self.vacc_agent_y %= self.height
            _observation = self._model.get_observation()

        elif action == ActionType.MOVE_E.value:
            """
            Handle MOVE_E action
            """
            self.vacc_agent_x += 1
            self.vacc_agent_x %= self.width
            _observation = self._model.get_observation()

        elif action == ActionType.MOVE_S.value:
            """
            Handle MOVE_S action
            """
            self.vacc_agent_y += 1
            self.vacc_agent_y %= self.height
            _observation = self._model.get_observation()

        elif action == ActionType.MOVE_W.value:
            """
            Handle MOVE_W action
            """
            self.vacc_agent_x -= 1
            self.vacc_agent_x %= self.width
            _observation = self._model.get_observation()

        return _observation, response


if __name__ == "__main__":

    np.random.seed(100)
    render = "simple" #"ansi"  # "ansi"  # change to "human"
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
        simulation_single_tick=True,
        debug=True,
        seed=0,
    )
    env = FreeExplorationEnv(config=env_config)
    print("USE RENDERER ?", env.use_renderer)
    record = True
    if record:
        # records the the rendering in the `recording` folder
        env = wrappers.Monitor(env, "recording", force=True)
    observation = env.reset()

    # import pdb; pdb.set_trace()
    done = False
    k = 0
    if not record:
        env.render(mode=render)
    donecount = 0
    while donecount < 3:
        print(
            """
        Valid Actions :
            MOVE_N = 0
            MOVE_E = 1
            MOVE_W = 2
            MOVE_S = 3

            VACCINATE = 4
            SIM_TICK = 5
        """
        )
        # _action = int(input("Enter action - ex : "))
        env.action_space.seed(k)
        _action = env.action_space.sample()
        # _action = 1

        print("Action : ", _action, "     Step:", k)
        observation, reward, done, info = env.step(_action)
        print(observation.shape)

        if not record:
            env.render(mode=render)
        print("Vacc_agent_location : ", env.vacc_agent_x, env.vacc_agent_y)
        k += 1
        print("=" * 100)
        if done: 
            donecount += 1
            observation = env.reset()
        # print(observation.shape)
        # print(k, reward, done)
    print(np.sum(observation, axis=0))
    print(info)
