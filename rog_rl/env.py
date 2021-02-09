import gym
from gym import spaces, wrappers
from gym.utils import seeding

from enum import Enum
import numpy as np

from rog_rl.agent_state import AgentState


class ActionType(Enum):
    STEP = 0
    VACCINATE = 1


class RogSimBaseEnv(gym.Env):

    def __init__(self, config={}):
        # Setup Config
        self.default_config = dict(
            width=50,
            height=50,
            population_density=0.75,
            vaccine_density=0.05,
            initial_infection_fraction=0.1,
            initial_vaccination_fraction=0.05,
            prob_infection=0.2,
            prob_agent_movement=0.0,
            disease_planner_config={
                "incubation_period_mu": 0,
                "incubation_period_sigma": 0,
                "recovery_period_mu": 20,
                "recovery_period_sigma": 0,
            },
            only_count_successful_vaccines=False,
            vaccine_score_weight=-1,
            use_np_model=True,
            max_simulation_timesteps=200,
            early_stopping_patience=20,
            use_renderer=False,  # can be "simple", "ansi"
            toric=True,
            fast_complete_simulation=True,
            fast_forward=False,
            dummy_simulation=False,
            debug=False)
        self.config = {}
        self.config.update(self.default_config)
        self.update_configs(config)

        self.seed(self.config.get('seed'))

        self.dummy_simulation = self.config["dummy_simulation"]
        self.debug = self.config["debug"]

        self.width = self.config["width"]
        self.height = self.config["height"]

        self.use_renderer = self.config["use_renderer"]
        self.vaccine_score_weight = self.config["vaccine_score_weight"]

        self.set_agent_state()
        self.set_action_type()

        self.disease_model = self.set_disease_model()

        self.action_space = self.set_action_space()
        self.observation_space = self.set_observation_space()

        self._model = None
        self.running_score = None
        self.np_random = np.random

        self.last_action = None
        self.last_action_response = None

        assert self.config['use_np_model'], "Non np model is not \
        , use use_np_model: True"

        self.renderer = False

        if self.use_renderer:
            self.initialize_renderer(mode=self.use_renderer)

        self.cumulative_reward = 0

    def update_configs(self, config={}):
        self.config.update(config)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_observation_space(self):
        return spaces.Box(low=np.uint8(0),
                          high=np.uint8(1),
                          shape=(self.width,
                                 self.height,
                                 len(AgentState)))

    def set_action_space(self):
        return spaces.MultiDiscrete(
            [
                len(ActionType), self.width, self.height
            ])

    def set_agent_state(self):
        self.agent_state = AgentState

    def set_action_type(self):
        self.action_type = ActionType

    def set_disease_model(self):
        if self.config['use_np_model']:
            from rog_rl.model_np import DiseaseSimModel
        else:
            from rog_rl.model import DiseaseSimModel
        return DiseaseSimModel

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

        self._step_count = 0 
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
        fast_complete_simulation = self.config['fast_complete_simulation']

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
            fast_complete_simulation,
            toric, seed=_simulator_instance_seed
        )

        self.env_reset()
        # Set the max timesteps of an env as the sum of :
        # - max_simulation_timesteps
        # - Number of Vaccines available

        self._max_episode_steps = self.config.get('max_episode_steps', None)
        if self._max_episode_steps is None:
            self._max_episode_steps = self.config['max_simulation_timesteps'] + \
                                        self._model.n_vaccines

#         Tick model
        if not self.config["use_np_model"]:
            # Not needed for model_np
            self._model.tick(self.config.get('fast_forward', False))

        if self.vaccine_score_weight < 0:
            self.running_score = self.get_current_game_score(
                include_vaccine_score=False)
        else:
            self.running_score = self.get_current_game_score(
                include_vaccine_score=True)
        self.cumulative_reward = 0
        observation = self.get_observation()
        return observation

    def env_reset(self):
        # Initialize location of vaccination agent
        pass

    def get_observation(self):
        obs = self._model.get_observation()
        return self.post_process_observation(obs)

    def post_process_observation(self, observation):
        return observation

    def get_action_response(self):
        return self.last_action_response

    def initialize_renderer(self, mode="human"):

        if self.use_renderer in ["simple"]:
            self.metadata = {'render.modes': ['simple', 'rgb_array'],
                             'video.frames_per_second': 5}
            from rog_rl.renderer import SimpleRenderer
            self.renderer = SimpleRenderer(
                grid_size=(self.width, self.height)
            )

        elif mode in ["human", "rgb_array"]:
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
            Initialize PIL Headless Renderer here
            for visualising during training
            """
            self.metadata = {'render.modes': ['PIL', 'rgb_array'],
                             'video.frames_per_second': 5}
            from rog_rl.renderer import PILRenderer
            self.renderer = PILRenderer(grid_size=(self.width, self.height))

        else:
            print("Invalid Mode selected for render:", mode)

        self.renderer.setup(mode=mode)

    def get_agents_by_state(self, state):
        if self.config['use_np_model']:
            obs = self._model.observation
            states = np.argmax(obs, axis=-1)
            idx = np.where(states == state.value)
            return [i for i in zip(idx[0], idx[1])]
        else:
            scheduler = self._model.get_scheduler()
            return scheduler.get_agents_by_state(state)

    def get_agents_grid(self):
        if self.config['use_np_model']:
            obs = self._model.observation
            return np.argmax(obs, axis=-1)
        else:
            scheduler = self._model.get_scheduler()
            return scheduler.get_agents_by_state()

    def get_agent_positions(self, agent):
        if self.config['use_np_model']:
            agent_x, agent_y = agent
            return agent_x, agent_y
        else:
            agent_x, agent_y = agent.pos
            return agent_x, agent_y

    def update_renderer(self, mode='human'):
        """
        Updates the latest board state on the renderer
        """
        # Draw Renderer
        # Update Renderer State
        model = self._model
        if self.config['use_np_model']:
            total_agents = model.n_agents
            _simulation_steps = model.schedule_steps
        else:
            scheduler = model.get_scheduler()
            total_agents = scheduler.get_agent_count()
            _simulation_steps = int(scheduler.steps)

        state_metrics = self.get_current_game_metrics()

        initial_vaccines = int(
            model.initial_vaccination_fraction * model.n_agents)

        _vaccines_given = \
            model.max_vaccines - model.n_vaccines - initial_vaccines

        # Game Steps includes steps in which each agent is vaccinated
        _game_steps = _simulation_steps + _vaccines_given

        self.renderer.update_stats(
            "SCORE",
            "{:.3f}".format(self.cumulative_reward))
        self.renderer.update_stats("VACCINE_BUDGET", "{}".format(
            model.n_vaccines))
        self.renderer.update_stats("SIMULATION_TICKS", "{}".format(
            _simulation_steps))
        self.renderer.update_stats("GAME_TICKS", "{}".format(_game_steps))

        self.update_env_renderer_stats()

        if self.use_renderer == 'simple':
            for key in state_metrics:
                self.renderer.update_stats(key, state_metrics[key])
            obs = self._model.get_observation()
            return self.renderer.get_render_output(obs)

        elif self.use_renderer == 'ansi':
            grid = self.get_agents_grid()
            render_output = self.renderer.render(self.width, self.height, grid)
            if self.debug:
                print(render_output)
            return render_output

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
                agents = self.get_agents_by_state(_state)
                for _agent in agents:
                    _agent_x, _agent_y = self.get_agent_positions(_agent)
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

    def update_env_renderer_stats(self):
        pass

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
#         _d["R0/10"] = self._model.contact_network.compute_R0()/10.0
        return _d

    def calculate_rewards(self):
        # Compute difference in game score
        if self.vaccine_score_weight < 0:
            current_score = self.get_current_game_score(
                include_vaccine_score=False)
            _step_reward = current_score - self.running_score
            self.cumulative_reward += _step_reward
            self.running_score = current_score
        else:
            current_score = self.get_current_game_score(
                include_vaccine_score=True)
            _step_reward = current_score - self.running_score
            self.running_score = current_score
            _done = not self._model.is_running()
            if _done:
                _step_reward = self.terminal_reward(
                    current_score, _step_reward)
            self.cumulative_reward += _step_reward

        return _step_reward

    def terminal_reward(self, current_score, _step_reward):
        susecptible_percentage = self.get_current_game_score(
            include_vaccine_score=False)
        _step_reward -= (current_score - susecptible_percentage) * \
            self.vaccine_score_weight
        return _step_reward

    def step_action(self, action):

        _observation = False

        response = "STEP"

        return _observation, response

    def step(self, action):
        # Handle dummy_simulation Mode
        if self.dummy_simulation:
            return self.dummy_env_step()

        assert self.action_space.contains(
            action), "%r (%s) invalid" % (action, type(action))
        if self._model is None:
            raise Exception("env.step() called before calling env.reset()")

        _done = False
        _info = {}

        _observation, response = self.step_action(action)
        if self._step_count >= self._max_episode_steps:
            self._model.run_simulation_to_end()
            _observation = self._model.get_observation()
        self._step_count += 1

        self.last_action = action
        self.last_action_response = response

        _step_reward = self.calculate_rewards()

        # Add custom game metrics to info key
        game_metrics = self.get_current_game_metrics()
        for _key in game_metrics.keys():
            _info[_key] = game_metrics[_key]

        _info['cumulative_reward'] = self.cumulative_reward
        _done = not self._model.is_running()

        _observation = self.post_process_observation(_observation)
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
            if hasattr(self.renderer, 'close'):
                self.renderer.close()
            self.renderer = False
        if self._model:
            # Delete the model instance if it exists
            self._model = None


if __name__ == "__main__":

    render = "simple"  # "ansi"  # change to "human"
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
            "incubation_period_sigma":  0,
            "recovery_period_mu": 20,
            "recovery_period_sigma":  0,
        },
        max_simulation_timesteps=200,
        early_stopping_patience=20,
        use_renderer=render,
        use_model_np=True,
        fast_complete_simuation=True,
        toric=False,
        dummy_simulation=False,
        debug=True)
    env = RogSimBaseEnv(config=env_config)
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
        _action = env.action_space.sample()
        # _action = input("Enter action - ex: [1, 4, 2] : ")
        # if _action.strip() == "":
        #     _action = env.action_space.sample()
        # else:
        #     _action = [int(x) for x in _action.split()]
        #     assert _action[0] in [0, 1]
        #     assert _action[1] in list(range(env._model.width))
        #     assert _action[2] in list(range(env._model.height))
        print("Action : ", _action)
        observation, reward, done, info = env.step(_action)
        if not record:
            env.render(mode=render)
        k += 1

        # print(observation.shape)
        # print(k, reward, done)
    # print(observation.shape())
