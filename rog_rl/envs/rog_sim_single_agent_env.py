import gym
from gym import spaces, wrappers
from gym.utils import seeding

from enum import Enum
import numpy as np


from rog_rl.agent_state import AgentState
# from rog_rl.model import DiseaseSimModel
# from rog_rl.model_np import DiseaseSimModel
from rog_rl.vaccination_response import VaccinationResponse

class ActionType(Enum):
    MOVE_N = 0
    MOVE_E = 1
    MOVE_W = 2
    MOVE_S = 3
    VACCINATE = 4
    SIM_TICK = 5


class RogSimSingleAgentEnv(gym.Env):
    """
    A single agent env contains a single
    vaccination agent which can move around the grid
    and apart from choose to vaccinate any cell
    or
    step ahead in the internal disease simulator.
    """

    def __init__(self, config={}):
        # Setup Config
        self.default_config = dict(
                    width=20,
                    height=20,
                    population_density=0.75,
                    vaccine_density=0.05,
                    initial_infection_fraction=0.1,
                    initial_vaccination_fraction=0.05,
                    prob_infection=0.2,
                    prob_agent_movement=0.0,
                    disease_planner_config={
                        "latent_period_mu": 2 * 4,
                        "latent_period_sigma": 0,
                        "incubation_period_mu": 5 * 4,
                        "incubation_period_sigma": 0,
                        "recovery_period_mu": 14 * 4,
                        "recovery_period_sigma": 0,
                    },
                    only_count_successful_vaccines=False,
                    vaccine_score_weight=-1,
		    use_np_model=False,
                    max_simulation_timesteps=20 * 20 * 10,
                    early_stopping_patience=14,
                    use_renderer=False,  # can be "human", "ansi"
                    toric=True,
                    dummy_simulation=False,
                    debug=False)
        self.config = {}
        self.config.update(self.default_config)
        self.config.update(config)

        self.dummy_simulation = self.config["dummy_simulation"]
        self.debug = self.config["debug"]

        self.width = self.config["width"]
        self.height = self.config["height"]

        self.use_renderer = self.config["use_renderer"]
        self.vaccine_score_weight = self.config["vaccine_score_weight"]
        
        if self.config['use_np_model']:
            from rog_rl.model_np import DiseaseSimModel
        else:
            from rog_rl.model import DiseaseSimModel
        self.disease_model = DiseaseSimModel
            

        """
        The action space is composed of 5 discrete actions :

        MOVE_N : Moves the vaccination-agent north
        MOVE_E : Moves the vaccination-agent east
        MOVE_W : Moves the vaccination-agent west
        MOVE_S : Moves the vaccination-agent south

        VACCINATE : Vaccinates the current location of the vaccination-agent
        SIM_TICK : adds a simulation tick to the disease model
        """
        self.action_space = spaces.Discrete(
            len(ActionType)
        )

        # In case we club the Exposed, Symptomatic, Infectious, and Vaccinated

        """
        The observation space in this case will be of shape (width, height, 3)
        where we represent 4 channels of information across the grid

        Channel 1 : Is the cell Susceptible
        Channel 2 : Is an agent in this cell exposed/infected at some point
        Channel 3 : Is an agent in this cell Vaccinated
        Channel 4 : Is the vaccination agent here
        """
        self.observation_channels = 4
        self.observation_space = spaces.Box(
                                    low=np.float32(0),
                                    high=np.float32(1),
                                    shape=(
                                        self.width,
                                        self.height,
                                        self.observation_channels))

        self._model = None
        self.running_score = None
        self.np_random = np.random

        self.renderer = False

        if self.use_renderer:
            self.initialize_renderer(mode=self.use_renderer)

        self.cumulative_reward = 0

        self.vacc_agent_x = 0
        self.vacc_agent_y = 0

        self._game_steps = 0

    def set_renderer(self, renderer):
        self.use_renderer = renderer
        if self.use_renderer:
            self.initialize_renderer(mode=self.use_renderer)

    def reset(self):
        # Delete Model if already exists
        if self._model:
            del self._model

        if self.dummy_simulation:
            """
            In dummy simulation mode
            return a randomly sampled observation
            """
            return self.observation_space.sample()

        width = self.config['width']
        height = self.config['height']
        population_density = self.config['population_density']
        vaccine_density = self.config['vaccine_density']
        initial_infection_fraction = self.config['initial_infection_fraction']
        initial_vaccination_fraction = \
            self.config['initial_vaccination_fraction']
        prob_infection = self.config['prob_infection']
        prob_agent_movement = self.config['prob_agent_movement']
        disease_planner_config = self.config['disease_planner_config']
        max_simulation_timesteps = self.config['max_simulation_timesteps']
        only_count_successful_vaccines = \
            self.config['only_count_successful_vaccines']
        early_stopping_patience = \
            self.config['early_stopping_patience']
        toric = self.config['toric']

        """
        Seeding Strategy :
            - The env maintains a custom seed/unsseded np.random instance
            accessible at self.np_random

            whenever env.seed() is called, the said np_random instance
            is seeded

            and during every new instantiation of a DiseaseEngine instance,
            it is seeded with a random number sampled from the self.np_random.
        """
        _simulator_instance_seed = self.np_random.randint(4294967296)
        # Instantiate Disease Model
        self._model = self.disease_model(
            width, height,
            population_density, vaccine_density,
            initial_infection_fraction, initial_vaccination_fraction,
            prob_infection, prob_agent_movement,
            disease_planner_config,
            max_simulation_timesteps, early_stopping_patience,
            only_count_successful_vaccines,
            toric, seed=_simulator_instance_seed
        )

        # Initialize location of vaccination agent
        self.vacc_agent_x = self.np_random.randint(self.width)
        self.vacc_agent_y = self.np_random.randint(self.height)

        self._game_steps = 1
        # Set the max timesteps of an env as the sum of :
        # - max_simulation_timesteps
        # - Number of Vaccines available

        self._max_episode_steps = self.config['max_simulation_timesteps'] + \
            self._model.n_vaccines

#         Tick model
        if not self.config["use_np_model"]:
            self._model.tick() # Not needed for model_np

        if self.vaccine_score_weight < 0:
            self.running_score = self.get_current_game_score(include_vaccine_score=False)
        else:
            self.running_score = self.get_current_game_score(include_vaccine_score=True)
        self.cumulative_reward = 0
        # return observation

        _observation = self._model.get_observation()

        _observation = self._post_process_observation(_observation)
        return _observation

    def _post_process_observation(self, observation):
        """
        Channel 1 : Is the cell Susceptible
        Channel 2 : Is an agent in this cell exposed/infected at some point
        Channel 3 : Is an agent in this cell Vaccinated
        Channel 4 : Is the vaccination agent here

        """
        vaccination_agent_channel = np.zeros((self.width, self.height))
        vaccination_agent_channel[self.vacc_agent_x, self.vacc_agent_y] = 1

        INFECTED_CHANNEL = observation[... , AgentState.EXPOSED.value] + \
            observation[... , AgentState.SYMPTOMATIC.value] + \
            observation[... , AgentState.INFECTIOUS.value] + \
            observation[... , AgentState.RECOVERED.value]

        p_obs = np.array([
            observation[... , AgentState.SUSCEPTIBLE.value],
            INFECTED_CHANNEL,
            observation[... , AgentState.VACCINATED.value],
            vaccination_agent_channel]
        ).T

        return p_obs

    def initialize_renderer(self, mode="human"):
        if mode in ["human", "rgb_array"]:
            self.metadata = {'render.modes': ['human', 'rgb_array'],
                             'video.frames_per_second': 5}
            from rog_rl.renderer import Renderer

            self.renderer = Renderer(
                    grid_size=(self.width, self.height)
                )
        elif mode in ["ansi"]:
            """
            Initialize ANSI Renderer here
            """
            self.metadata = {'render.modes': ['human', 'ansi'],
                             'video.frames_per_second': 5}
            from rog_rl.renderer import ANSIRenderer
            self.renderer = ANSIRenderer()

        elif mode in ["PIL"]:
            """
            Initialize PIL Headless Renderer here for visualising during training
            """
            self.metadata = {'render.modes': ['PIL'],
                             'video.frames_per_second': 5}
            from rog_rl.renderer import PILRenderer
            self.renderer = PILRenderer(grid_size=(self.width, self.height))
        else:
            print("Invalid Mode selected for render:",mode)

        self.renderer.setup(mode=mode)

    def update_renderer(self, mode='human'):
        """
        Updates the latest board state on the renderer
        """
        # Draw Renderer
        # Update Renderer State
        model = self._model
        scheduler = model.get_scheduler()
        total_agents = scheduler.get_agent_count()
        state_metrics = self.get_current_game_metrics()

        initial_vaccines = int(
            model.initial_vaccination_fraction * model.n_agents)

        _vaccines_given = \
            model.max_vaccines - model.n_vaccines - initial_vaccines

        _simulation_steps = int(scheduler.steps)

        # Game Steps includes steps in which each agent is vaccinated
        _game_steps = _simulation_steps + _vaccines_given

        self.renderer.update_stats(
                    "SCORE",
                    "{:.3f}".format(self.cumulative_reward))
        self.renderer.update_stats("VACCINE_BUDGET", "{}".format(
            model.n_vaccines))
        self.renderer.update_stats("SIMULATION_TICKS", "{}".format(
            _simulation_steps))
        self.renderer.update_stats("GAME_TICKS", "{}".format(self._game_steps))

        # Add VACC_AGENT coords to render state
        self.renderer.update_stats(
            "VACC_AGENT_X",
            str(self.vacc_agent_x)
        )
        self.renderer.update_stats(
            "VACC_AGENT_Y",
            str(self.vacc_agent_y)
        )


        for _state in AgentState:
            key = "population.{}".format(_state.name)
            stats = state_metrics[key]
            self.renderer.update_stats(
                key,
                "{} ({:.2f}%)".format(
                    int(stats * total_agents),
                    stats*100
                )
            )
            if mode in ["human", "rgb_array"]:
                color = self.renderer.COLOR_MAP.get_color(_state)
                agents = scheduler.get_agents_by_state(_state)
                for _agent in agents:
                    _agent_x, _agent_y = _agent.pos
                    self.renderer.draw_cell(
                                _agent_x, _agent_y,
                                color
                            )
        if mode in ["human", "rgb_array"]:
            # Update the rest of the renderer
            self.renderer.pre_render()

            # Only in case of recording via Monitor or setting mode = rgb_array
            # we require the rgb image
            if isinstance(self, wrappers.Monitor):
                return_rgb_array = mode in ["human", "rgb_array"]
            else:
                return_rgb_array = mode == "rgb_array"
            render_output = self.renderer.post_render(return_rgb_array)
            return render_output
        elif mode == "ansi":
            render_output = self.renderer.render(self._model.grid)
            if self.debug:
                print(render_output)
            return render_output

    def get_current_game_score(self, include_vaccine_score):
        """
        Returns the current game score

        The game score is currently represented as :
            (percentage of susceptibles left in the population)
        """
        score = self._model.get_population_fraction_by_state(
                    AgentState.SUSCEPTIBLE)
        if include_vaccine_score:
            score += self._model.get_population_fraction_by_state(
                        AgentState.VACCINATED)

        return score

    def get_current_game_metrics(self, dummy_simulation=False):
        """
        Returns a dictionary containing important game metrics
        """
        _d = {}
        # current population fraction of different states
        for _state in AgentState:
            if not dummy_simulation:
                _value = self._model.get_population_fraction_by_state(_state)
            else:
                _value = self.np_random.rand()

            _key = "population.{}".format(_state.name)
            _d[_key] = _value
        # Add "Protected" and "Affected"
        _d["population.PROTECTED"] = _d["population.SUSCEPTIBLE"] + \
                                     _d["population.VACCINATED"]
        _d["population.AFFECTED"] = 1. - _d["population.PROTECTED"]
        # Add R0 to the game metrics
        # _d["R0/10"] = self._model.contact_network.compute_R0()/10.0
        return _d

    def step(self, action):
        # Handle dummy_simulation Mode
        if self.dummy_simulation:
            return self.dummy_env_step()

        assert self.action_space.contains(
            action), "%r (%s) invalid" % (action, type(action))
        if self._model is None:
            raise Exception("env.step() called before calling env.reset()")

        if self.debug:
            print("Action : ", action)

        _observation = False
        _done = False
        _info = {}

        if action == ActionType.SIM_TICK.value:
            """
            Handle SIM_TICK action
            """
            # Handle action propagation in real simulator
            self._model.tick()
            _observation = self._model.get_observation()
        elif action == ActionType.VACCINATE.value:
            """
            Handle VACCINATE action
            """
            # Vaccinate the cell where the vaccination agent currently is
            cell_x = self.vacc_agent_x
            cell_y = self.vacc_agent_y
            vaccination_success, response = \
                self._model.vaccinate_cell(cell_x, cell_y)
            _observation = self._model.get_observation()

            # Force Run simulation to completion if
            # run out of vaccines
            if response == VaccinationResponse.AGENT_VACCINES_EXHAUSTED:
                while self._model.is_running():
                    self._model.tick()
                    _observation = self._model.get_observation()

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

        _done = not self._model.is_running()

        self._game_steps += 1
        if self._game_steps > self._max_episode_steps:
            # When we run out of game steps
            # ensure that the sim model runs till completion.
            _done = True
            while self._model.is_running():
                self._model.tick()
                _observation = self._model.get_observation()

        # Compute difference in game score

        if self.vaccine_score_weight < 0:
            current_score = self.get_current_game_score(include_vaccine_score=False)
            _step_reward = current_score - self.running_score
            self.cumulative_reward += _step_reward
            self.running_score = current_score
        else:
            current_score = self.get_current_game_score(include_vaccine_score=True)
            _step_reward = current_score - self.running_score
            self.running_score = current_score
            _done = not self._model.is_running()
            if _done:
                susecptible_percentage = self.get_current_game_score(include_vaccine_score=False)
                _step_reward -= (current_score - susecptible_percentage) * self.vaccine_score_weight
            self.cumulative_reward += _step_reward

        # Add custom game metrics to info key
        game_metrics = self.get_current_game_metrics()
        for _key in game_metrics.keys():
            _info[_key] = game_metrics[_key]

        _info['cumulative_reward'] = self.cumulative_reward

        _observation = self._post_process_observation(_observation)
        return _observation, _step_reward, _done, _info

    def dummy_env_step(self):
        """
        Implements a fake env.step for faster Integration Testing
        with RL experiments framework
        """
        observation = self.observation_space.sample()
        reward = self.np_random.rand()
        done = True if self.np_random.rand() < 0.01 else False
        info = {}
        game_metrics = self.get_current_game_metrics(dummy_simulation=True)
        info.update(game_metrics)

        return observation, reward, done, info

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        """
        This methods provides the option to render the
        environment's behavior to a window which should be
        readable to the human eye if mode is set to 'human'.
        """
        if not self.use_renderer:
            return

        if not self.renderer:
            self.initialize_renderer(mode=mode)

        return self.update_renderer(mode=mode)

    def close(self):
        if self.renderer:
            self.renderer.close()
            self.renderer = False
        if self._model:
            # Delete the model instance if it exists
            self._model = None


if __name__ == "__main__":

    render = "PIL" # "ansi"  # change to "human"
    env_config = dict(
                    width=3,
                    height=6,
                    population_density=1.0,
                    vaccine_density=1.0,
                    initial_infection_fraction=0.04,
                    initial_vaccination_fraction=0,
                    prob_infection=0.2,
                    prob_agent_movement=0.0,
                    disease_planner_config={
                        "latent_period_mu": 2 * 4,
                        "latent_period_sigma": 0,
                        "incubation_period_mu": 5 * 4,
                        "incubation_period_sigma": 0,
                        "recovery_period_mu": 14 * 4,
                        "recovery_period_sigma": 0,
                    },
                    vaccine_score_weight=0.5,
                    max_simulation_timesteps=20 * 20 * 10,
                    early_stopping_patience=14,
                    use_renderer=render,  # can be "human", "ansi"
                    toric=False,
                    dummy_simulation=False,
                    debug=True)
    env = RogSimSingleAgentEnv(config=env_config)
    print("USE RENDERER ?", env.use_renderer)
    record = True
    if record:
        # records the the rendering in the `recording` folder
        env = wrappers.Monitor(env, "recording", force=True)
    observation = env.reset()
    done = False
    k = 0
    env.render(mode=render)
    while not done:
        print("""
        Valid Actions :
            MOVE_N = 0
            MOVE_E = 1
            MOVE_W = 2
            MOVE_S = 3

            VACCINATE = 4
            SIM_TICK = 5
        """)
        # _action = int(input("Enter action - ex : "))
        _action = env.action_space.sample()

        print("Action : ", _action)
        observation, reward, done, info = env.step(_action)
        print(observation.shape)
        env.render(mode=render)
        print("Vacc_agent_location : ", env.vacc_agent_x, env.vacc_agent_y)
        k += 1
        print("="*100)
        # print(observation.shape)
        # print(k, reward, done)
