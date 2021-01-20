import numpy as np

from rog_rl.agent_state import AgentState
from rog_rl.vaccination_response import VaccinationResponse
from scipy.stats import truncnorm
from scipy.signal import convolve2d
from collections import deque
from skimage.measure import label as connected_components


class DiseaseSimModel:
    """
    The model class holds the model-level attributes, manages the agents,
    and generally handles
    the global level of our model.

    There is only one model-level parameter: how many agents
    the model contains. When a new model
    is started, we want it to populate itself with the given number of agents.

    The scheduler is a special model component which controls the order
    in which agents are activated.
    """

    def __init__(
        self,
        width=50,
        height=50,
        population_density=0.75,
        vaccine_density=0,
        initial_infection_fraction=0.1,
        initial_vaccination_fraction=0.00,
        prob_infection=0.2,
        prob_agent_movement=0.0,
        disease_planner_config={
            "incubation_period_mu": 0,
            "incubation_period_sigma":  0,
            "recovery_period_mu": 20,
            "recovery_period_sigma":  0,
        },
        max_timesteps=200,
        early_stopping_patience=20,
        only_count_successful_vaccines=True,
        fast_complete_simulation=True,
        toric=True,
        seed=None
    ):
        self.width = width
        self.height = height
        # fraction of the whole grid that is initiailized with agents
        self.population_density = population_density
        self.vaccine_density = vaccine_density

        self.n_agents = self.width * self.height
        self.n_vaccines = int(self.n_agents * self.vaccine_density)

        self.initial_infection_fraction = initial_infection_fraction
        self.initial_vaccination_fraction = initial_vaccination_fraction

        self.prob_infection = prob_infection
        self.prob_agent_movement = prob_agent_movement

        self.disease_planner_config = disease_planner_config

        self.max_timesteps = max_timesteps
        self.early_stopping_patience = early_stopping_patience
        self.toric = toric
        self.boundary = 'wrap' if toric else 'fill'
        self.seed = seed
        self.only_count_successful_vaccines = only_count_successful_vaccines

        self.last_n_susceptible_fractions = deque(
            maxlen=early_stopping_patience)

        self.rng = np.random.RandomState(seed=self.seed)
        self.gridshape = (self.width, self.height)
        self.neighbor_kernel_r1 = np.ones((3, 3))
        self.neighbor_kernel_r1[1, 1] = 0

        self.fast_complete_simulation = fast_complete_simulation and not toric

        self.initialize_np()

        self.running = True

    ###########################################################################
    ###########################################################################
    # Setup Initialization Helper Functions
    ###########################################################################

    def initialize_np(self):

        # Initialize observation
        self.observation = np.zeros(
            (*self.gridshape, len(AgentState)), np.uint8)
        self.observation[..., AgentState.SUSCEPTIBLE.value] = 1

        # Schedule
        self.infection_scheduled_grid = np.zeros(self.gridshape, dtype=np.bool)
        self.infection_base_time_grid = np.zeros(
            self.gridshape, dtype=np.int32) + np.NINF
        self.schedule_steps = 0

        # Initialize time values
        incubation_period_mu = \
            self.disease_planner_config["incubation_period_mu"]
        incubation_period_sigma = \
            self.disease_planner_config["incubation_period_sigma"]
        recovery_period_mu = \
            self.disease_planner_config["recovery_period_mu"]
        recovery_period_sigma = \
            self.disease_planner_config["recovery_period_sigma"]

        exposure_tvals = np.zeros(self.gridshape)
        self.incubation_tvals = self._sample_tvals(incubation_period_mu,
                                                   incubation_period_sigma,
                                                   exposure_tvals)
        self.recovery_tvals = self._sample_tvals(recovery_period_mu,
                                                 recovery_period_sigma,
                                                 self.incubation_tvals)
        self.transition_map = {AgentState.SUSCEPTIBLE: [AgentState.INFECTIOUS,
                                                        self.incubation_tvals],
                               AgentState.INFECTIOUS: [AgentState.RECOVERED,
                                                       self.recovery_tvals], }
        # Infect
        n_infect = int(self.initial_infection_fraction * self.n_agents)
        infect_list = self.rng.choice(np.arange(self.width * self.height),
                                      size=n_infect, replace=False)
        infect_locs = (infect_list // self.height, infect_list % self.height)
        self.infection_scheduled_grid[infect_locs] = True
        self.infection_base_time_grid[infect_locs] = self.schedule_steps
        self.observation[self.infection_scheduled_grid] = 0
        self.observation[self.infection_scheduled_grid,
                         AgentState.INFECTIOUS.value] = 1

        # Initial vaccinate  - Skipped for now - Ignore locations infected
        # and vaccinate in similar manner
        n_vaccine_init = int(self.initial_vaccination_fraction * self.n_agents)
        not_infected = list(
            set(np.arange(self.height * self.height)) - set(infect_list))
        vaccinate_list = self.rng.choice(
            not_infected, size=n_vaccine_init, replace=False)
        vaccinate_locs = (vaccinate_list // self.height,
                          vaccinate_list % self.height)
        vaccinate_mask = np.zeros(self.gridshape, dtype=np.bool)
        vaccinate_mask[vaccinate_locs] = True
        self.observation[vaccinate_mask] = 0
        self.observation[vaccinate_mask, AgentState.VACCINATED.value] = 1
        self.max_vaccines = self.n_vaccines + n_vaccine_init

    def _sample_tvals(self, mu, sigma, minvals):
        assert sigma >= 0
        if sigma == 0:
            t = np.maximum(minvals, np.zeros_like(minvals) + mu)
        elif sigma > 0:
            # Flatten to 1D as truncnorm.rvs only takes 1D
            minv = np.ravel(minvals)
            a = (minv - mu) / sigma
#             maxv = np.inf
#             b = (maxv - mu) / sigma
            b = np.zeros_like(a) + np.inf
            # truncnorm.rvs can draw from a 1D array of distributions -
            # But the size parameter doesn't work in multi distribution
            t = truncnorm.rvs(a, b, loc=mu, scale=sigma, random_state=self.rng)

        tvals = np.int32(t.reshape(self.gridshape))
        return tvals

    ###########################################################################
    ###########################################################################
    # State Aggregation
    #       - Functions for easy access/aggregation of simulation wide state
    ###########################################################################

    def get_observation(self):
        assert np.all(self.observation.sum(axis=-1) == 1)
        # Assertion disabled for perf reasons
        return self.observation

    def get_population_fraction_by_state(self, state: AgentState):
        return np.sum(self.observation[..., state.value]) / self.n_agents

    def is_running(self):
        return self.running

    ###########################################################################
    ###########################################################################
    # Actions
    #        - Functions for actions that can be performed on the model
    ###########################################################################

    def step(self):
        """
        A model step. Used for collecting data and advancing the schedule
        """
        if not self.running:
            return
        self.schedule_steps += 1
        self.propagate_infections_np()
        self.last_n_susceptible_fractions.append(
            self.get_population_fraction_by_state(
                AgentState.SUSCEPTIBLE))
        self.step_schedule()
        self.simulation_completion_checks()

    def vaccinate_cell(self, cell_x, cell_y):
        """
        Vaccinates an agent at cell_x, cell_y, if present

        Response with :
        (is_vaccination_successful, vaccination_response)
        of types
        (boolean, VaccinationResponse)
        """
        if not self.running:
            return False, VaccinationResponse.SIMULATION_NOT_RUNNING

        # Case 0 : No vaccines left
        if self.n_vaccines <= 0:
            return False, VaccinationResponse.AGENT_VACCINES_EXHAUSTED

        if not self.only_count_successful_vaccines:
            self.n_vaccines -= 1

        success = False

        agent_state = np.argmax(self.observation[cell_x, cell_y])
        if agent_state == AgentState.SUSCEPTIBLE.value:
            # Case 2 : Agent is susceptible, and can be vaccinated
            if self.only_count_successful_vaccines:
                self.n_vaccines -= 1

            if self.n_vaccines >= 0:
                self.observation[cell_x, cell_y] = 0
                self.observation[cell_x, cell_y,
                                 AgentState.VACCINATED.value] = 1
                # Remove the scheduled infection if
                # vaccine is given before trigger
                self.infection_scheduled_grid[cell_x, cell_y] = False

            response = VaccinationResponse.VACCINATION_SUCCESS
            success = True
        elif agent_state == AgentState.INFECTIOUS.value:
            # Agent is already Infectious, its a waste of vaccination
            response = VaccinationResponse.AGENT_INFECTIOUS
        elif agent_state == AgentState.RECOVERED.value:
            # Agent is already Recovered, its a waste of vaccination
            response = VaccinationResponse.AGENT_RECOVERED
        elif agent_state == AgentState.VACCINATED.value:
            # Agent is already Vaccinated, its a waste of vaccination
            response = VaccinationResponse.AGENT_VACCINATED
        else:
            raise NotImplementedError()

        # If vaccines finished, run the simulation to end
        if self.n_vaccines == 0:
            self.run_simulation_to_end()

        return success, response
    ###########################################################################
    ###########################################################################
    # Misc
    ###########################################################################

    def simulation_completion_checks(self):
        """
        Simulation is complete if :
            - if the timesteps have exceeded the number of max_timesteps
            or
            - the fraction of susceptible population is <= 0
            or
            - the fraction of susceptible population has not changed since the
            last N timesteps
        """
        if self.schedule_steps > self.max_timesteps - 1:
            self.running = False
            return

        susceptible_population = self.get_population_fraction_by_state(
            AgentState.SUSCEPTIBLE)
        if susceptible_population <= 0:
            self.running = False
            return

        if self.schedule_steps > self.early_stopping_patience:
            if len(set(self.last_n_susceptible_fractions)) == 1:
                self.running = False
                return

    def run_simulation_to_end(self):
        """
        Finds the final state of the simulation
        and sets that as the observation
        Also sets self.running to True
        """
        if self.fast_complete_simulation:

            obs = self.observation.copy()
            sus = obs[..., AgentState.SUSCEPTIBLE.value]

            rec = obs[..., AgentState.RECOVERED.value]
            # Use the scheduled grid to find
            # current infections and scheduled ones
            sym = np.int32(self.infection_scheduled_grid) - rec
            assert np.all(sym >= 0)

            ss = sus + sym
            comps = connected_components(ss, background=0)
            for cval in range(1, np.max(comps)+1):
                match = (comps == cval)
                if np.any(sym[match]):
                    rec[match] = 1
                    sus[match] = 0

            self.observation[(rec == 1)] = 0
            self.observation[(rec == 1), AgentState.RECOVERED.value] = 1
            self.observation[(sus == 1)] = 0
            self.observation[(sus == 1), AgentState.SUSCEPTIBLE.value] = 1

            self.running = False
        else:
            while self.running:
                self.tick()

    def tick(self, fast_forward=False):
        """
        provides option for fast forwarding
        """
        if fast_forward:
            self.run_simulation_to_end()
        else:
            self.tick_once()

    def tick_once(self):
        """
        a mirror function for the internal step function
        to help avoid confusion in the RL codebases (with the RL step)
        """
        self.step()

    def step_schedule(self):
        for state, [nextstate, tvals] in self.transition_map.items():
            time_match = (self.infection_base_time_grid +
                          tvals) == self.schedule_steps
            state_match = self.observation[..., state.value]
            should_transition = np.logical_and(time_match, state_match)
            self.observation[should_transition, state.value] = 0
            self.observation[should_transition, nextstate.value] = 1

    def propagate_infections_np(self):

        # Infect neighbours
        infectious = self.observation[..., AgentState.INFECTIOUS.value]
        infected_neighbours = convolve2d(infectious, self.neighbor_kernel_r1,
                                         mode='same', boundary=self.boundary)
#         p = np.zeros(self.gridshape) + self.prob_infection
        p = self.prob_infection
        # GP Series if all prob_infection are same
        # --> p + p*(1-p) + p*(1-p)^2 + ... p*(1-p)^(n-1)
        infection_prob_allneighbours = 1-(1-p)**infected_neighbours
        infected = self.rng.rand(
            *self.gridshape,) < infection_prob_allneighbours
        already_infected = self.infection_scheduled_grid
        infected = np.logical_and(infected, ~already_infected)
        self.infection_scheduled_grid[infected] = True
        self.infection_base_time_grid[infected] = self.schedule_steps


if __name__ == "__main__":
    model = DiseaseSimModel(
        width=50,
        height=50,
        population_density=0.99,
        vaccine_density=0.0,
        initial_infection_fraction=0.99,
        initial_vaccination_fraction=0.0,
        prob_infection=1.0,
        prob_agent_movement=0.0,
        disease_planner_config={
            "incubation_period_mu":  5 * 4,
            "incubation_period_sigma":  0,
            "recovery_period_mu":  14 * 4,
            "recovery_period_sigma":  0,
        },
        max_timesteps=5,
        early_stopping_patience=14,
        toric=True)

    import time
    per_step_times = []
    for k in range(100):
        _time = time.time()
        model.step()
        per_step_times.append(time.time() - _time)
        _obs = model.get_observation()
        # print(model.get_population_fraction_by_state(AgentState.SUSCEPTIBLE))

        # Random Vaccinations
        # random_x = model.random.choice(range(50))
        # random_y = model.random.choice(range(50))
        # print(model.vaccinate_cell(random_x, random_y))

        # print(per_step_times[-1])
        # print(model.datacollector.get_model_vars_dataframe())
        # print("S", model.schedule.get_agent_count_by_state(AgentState.SUSCEPTIBLE))  # noqa
        # print("I", model.schedule.get_agent_count_by_state(AgentState.INFECTIOUS))  # noqa
        # print("R", model.schedule.get_agent_count_by_state(AgentState.RECOVERED))  # noqa
        # print(viz.render())
    per_step_times = np.array(per_step_times)
    print("Per Step Time : {} += {}", per_step_times.mean(), per_step_times.std())  # noqa
