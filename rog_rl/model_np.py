import numpy as np

from rog_rl.agent_state import AgentState
from scipy.stats import truncnorm
from scipy.signal import convolve2d
from collections import deque

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
            "latent_period_mu":  2 * 4,
            "latent_period_sigma":  0,
            "incubation_period_mu":  5 * 4,
            "incubation_period_sigma":  0,
            "recovery_period_mu":  14 * 4,
            "recovery_period_sigma":  0,
        },
        max_timesteps=200,
        early_stopping_patience=14,
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
        self.seed = seed
        
        self.last_n_susceptible_fractions = deque(maxlen=early_stopping_patience)
        
        self.rng = np.random.RandomState(seed=self.seed)
        self.gridshape = (self.width, self.height)
        self.neighbor_kernel_r1 = np.ones((3,3))
        self.neighbor_kernel_r1[1,1] = 0
        
        self.initialize_np()

        self.running = True

    ###########################################################################
    ###########################################################################
    # Setup Initialization Helper Functions
    ###########################################################################
    
    def initialize_np(self):
        
        # Initialize observation
        self.observation = np.zeros((*self.gridshape, len(AgentState)), np.uint8)
        self.observation[..., AgentState.SUSCEPTIBLE.value] = 1
                            
        # Schedule
        self.infection_scheduled_grid = np.zeros(self.gridshape, dtype=np.bool)
        self.infection_base_time_grid = np.zeros(self.gridshape, dtype=np.int32) + np.NINF
        self.schedule_steps = 0

        # Initialize time values
        latent_period_mu=self.disease_planner_config["latent_period_mu"]
        latent_period_sigma=self.disease_planner_config["latent_period_sigma"]
        incubation_period_mu=self.disease_planner_config["incubation_period_mu"]
        incubation_period_sigma=self.disease_planner_config["incubation_period_sigma"]
        recovery_period_mu=self.disease_planner_config["recovery_period_mu"]
        recovery_period_sigma=self.disease_planner_config["recovery_period_sigma"]
        
        self.exposure_tvals = np.zeros(self.gridshape)
        self.latent_tvals = self._sample_tvals(latent_period_mu,
                                              latent_period_sigma,
                                              self.exposure_tvals)
        self.incubation_tvals = self._sample_tvals(incubation_period_mu,
                                              incubation_period_sigma,
                                              self.latent_tvals)
        self.recovery_tvals = self._sample_tvals(recovery_period_mu,
                                              recovery_period_sigma,
                                              self.incubation_tvals)
        self.transition_map = {AgentState.SUSCEPTIBLE: [AgentState.EXPOSED, self.exposure_tvals],
                                AgentState.EXPOSED: [AgentState.INFECTIOUS, self.latent_tvals],
                                AgentState.INFECTIOUS: [AgentState.SYMPTOMATIC, self.incubation_tvals],
                                AgentState.SYMPTOMATIC: [AgentState.RECOVERED, self.recovery_tvals],}
        # Infect
        n_infect = int(self.initial_infection_fraction * self.n_agents)
        infect_list = self.rng.choice(np.arange(self.width * self.height), 
                                       size=n_infect, replace=False)       
        infect_locs = (infect_list // self.width, infect_list % self.height)
        self.infection_scheduled_grid[infect_locs] = True
        self.infection_base_time_grid[infect_locs] = self.schedule_steps
        self.observation[self.infection_scheduled_grid] = 0
        self.observation[self.infection_scheduled_grid, AgentState.EXPOSED.value] = 1
                             
        # Initial vaccinate  - Skipped for now - Ignore locations infected and vaccinate in similar manner
        n_vaccine_init = int(self.initial_vaccination_fraction * self.n_agents)
        not_infected = list(set(np.arange(self.width * self.height)) - set(infect_list))
        vaccinate_list = self.rng.choice(not_infected, size=n_vaccine_init, replace=False)
        vaccinate_locs = (vaccinate_list // self.width, vaccinate_list % self.height)
        vaccinate_mask = np.zeros(self.gridshape, dtype=np.bool)
        vaccinate_mask[vaccinate_locs] = True
        self.observation[vaccinate_mask] = 0
        self.observation[vaccinate_mask, AgentState.VACCINATED.value] = 1
        self.max_vaccines = self.n_vaccines + n_vaccine_init
                            
    def _sample_tvals(self, mu, sigma, minvals):
        minv = np.ravel(minvals) # Flatten to 1D as truncnorm.rvs only takes 1D
        if sigma > 0:
            a = (minv - mu) / sigma 
        else:
            a = (minv - mu) * np.inf
#         maxv = np.inf
#         b = (maxv - mu) / sigma
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
        assert  np.all(self.observation.sum(axis=-1) == 1)
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

        # Case 0 : No vaccines left
        if self.n_vaccines <= 0:
            return False, VaccinationResponse.AGENT_VACCINES_EXHAUSTED
        self.n_vaccines -= 1

        agent_state = np.argmax(self.observation[cell_x, cell_y])
        if agent_state == AgentState.SUSCEPTIBLE.value:
            # Case 2 : Agent is susceptible, and can be vaccinated
            self.observation[cell_x, cell_y] = 0
            self.observation[cell_x, cell_y, AgentState.VACCINATED.value] = 1
            return True, VaccinationResponse.VACCINATION_SUCCESS
        elif agent_state == AgentState.EXPOSED.value:
            # Case 3 : Agent is already exposed, and its a waste of vaccination
            return False, VaccinationResponse.AGENT_EXPOSED
        elif agent_state == AgentState.INFECTIOUS.value:
            # Case 4 : Agent is already infectious,
            # and its a waste of vaccination
            return False, VaccinationResponse.AGENT_INFECTIOUS
        elif agent_state == AgentState.SYMPTOMATIC.value:
            # Case 5 : Agent is already Symptomatic,
            # and its a waste of vaccination
            return False, VaccinationResponse.AGENT_SYMPTOMATIC
        elif agent_state == AgentState.RECOVERED.value:
            # Case 6 : Agent is already Recovered,
            # and its a waste of vaccination
            return False, VaccinationResponse.AGENT_RECOVERED
        elif agent_state == AgentState.VACCINATED.value:
            # Case 7 : Agent is already Vaccination,
            # and its a waste of vaccination
            return False, VaccinationResponse.AGENT_VACCINATED
        raise NotImplementedError()

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

    def tick(self):
        """
        a mirror function for the internal step function
        to help avoid confusion in the RL codebases (with the RL step)
        """
        self.step()
        
    def step_schedule(self):
        for state, [nextstate, tvals] in self.transition_map.items():
            time_match = (self.infection_base_time_grid + tvals) == self.schedule_steps
            state_match  = self.observation[..., state.value]
            should_transition = np.logical_and(time_match, state_match)
            self.observation[should_transition, state.value] = 0
            self.observation[should_transition, nextstate.value] = 1

    def propagate_infections_np(self):
        
        # Infect neighbours
        infectious = self.observation[..., AgentState.INFECTIOUS.value] + \
                     self.observation[..., AgentState.SYMPTOMATIC.value]
        infected_neighbours = convolve2d(infectious, self.neighbor_kernel_r1, 
                                         mode='same')
        p = np.zeros(self.gridshape) + self.prob_infection
        infection_prob_allneighbours = p * (1-p**infected_neighbours)/ (1-p) # GP Series if all prob_infection are same
        infected = self.rng.rand(*self.gridshape,) < infection_prob_allneighbours
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
            "latent_period_mu":  2 * 4,
            "latent_period_sigma":  0,
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
        # print("E", model.schedule.get_agent_count_by_state(AgentState.EXPOSED))  # noqa
        # print("I", model.schedule.get_agent_count_by_state(AgentState.INFECTIOUS))  # noqa
        # print("R", model.schedule.get_agent_count_by_state(AgentState.RECOVERED))  # noqa
        # print(viz.render())
    per_step_times = np.array(per_step_times)
    print("Per Step Time : {} += {}", per_step_times.mean(), per_step_times.std())  # noqa
