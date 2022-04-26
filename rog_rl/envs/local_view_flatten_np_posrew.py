from gym import spaces
from rog_rl import FreeExplorationEnv
import numpy as np
import gym


class LocalViewFlattenPosReward(FreeExplorationEnv):
    def __init__(self, config):
        self.local_radius = config.get("local_radius", 1)
        self.toric = config.get("toric")
        self.reward_vaccinated_weight = config.copy().pop("reward_vaccinated_weight")
        super().__init__(config)

    def set_observation_space(self):
        radius_dim = self.local_radius * 2 + 1
        return spaces.Box(
            low=np.uint8(0),
            high=np.uint8(1),
            shape=(radius_dim * radius_dim * len(self.agent_state),),
        )

    def update_env_renderer_stats(self):
        self.renderer.update_stats("LOCAL_RADIUS", str(self.local_radius))
        super().update_env_renderer_stats()

    def post_process_observation(self, observation):
        x, y = self.vacc_agent_x, self.vacc_agent_y
        local_radius = self.local_radius

        if self.toric:
            padded_obs = observation.copy()
        else:
            # Note - Padding with all 0s
            # Might want to also consider padding with "suscpetible"
            pad_axis = (local_radius, local_radius)
            pad_widths = (
                pad_axis,
                pad_axis,
                (0, 0),
            )
            padded_obs = np.pad(observation, pad_widths)
            x += local_radius
            y += local_radius

        x_min, x_max = x - local_radius, x + local_radius
        y_min, y_max = y - local_radius, y + local_radius
        rows = padded_obs.take(range(x_min, x_max + 1), axis=0, mode="wrap") # Mode wrap is just for toric
        final_obs = rows.take(range(y_min, y_max + 1), axis=1, mode="wrap")

        final_obs = final_obs.ravel()  # flatten

        return final_obs

    def step(self, action):
        observation, _, done, info = super().step(action)
        new_reward = 0
        if done:
            new_reward += (
                info["population.SUSCEPTIBLE"]
                + info["population.VACCINATED"] * self.reward_vaccinated_weight
            )
        return observation, new_reward, done, info

gym.envs.register(
     id='FreeExplorationEnvPosRew-v1', entry_point="local_view_flatten_np_posrew:LocalViewFlattenPosReward"
)

if __name__ == "__main__":

    np.random.seed(100)
    render = "simple" #"ansi"  # "ansi"  # change to "human"
    env_config = dict(
        width=15,
        height=15,
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
        max_simulation_timesteps=100,
        early_stopping_patience=20,
        use_renderer=render,  # can be "human", "ansi"
        use_np_model=True,
        toric=False,
        dummy_simulation=False,
        simulation_single_tick=True,
        debug=True,
        seed=0,
        local_radius=2,
        reward_vaccinated_weight=0,
    )
    env = LocalViewFlattenPosReward(config=env_config)
    print("USE RENDERER ?", env.use_renderer)
    record = True
    if record:
        # records the the rendering in the `recording` folder
        env = gym.wrappers.Monitor(env, "recording", force=True)
    observation = env.reset()

    # import pdb; pdb.set_trace()
    done = False
    k = 0
    if not record:
        env.render(mode=render)
    while not done:
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
        # _action = env.action_space.sample()
        _action = np.random.randint(5)
        # _action = 1

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