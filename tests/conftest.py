import pytest
from rog_rl import BaseGridRogRLEnv  # noqa
from rog_rl import FreeExplorationEnv  # noqa
from click.testing import CliRunner


@pytest.fixture(scope="module")
def runner():
    return CliRunner()


@pytest.mark.skip(reason="legacy mesa model is not supported")
# @pytest.fixture(scope="module")
def all_mesa_envs():
    render = "simple"  # "ansi"  # change to "PIL"
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
            "latent_period_mu": 2 * 4,
            "latent_period_sigma": 0,
            "incubation_period_mu": 5 * 4,
            "incubation_period_sigma": 0,
            "recovery_period_mu": 14 * 4,
            "recovery_period_sigma": 0,
        },
        max_simulation_timesteps=200,
        early_stopping_patience=14,
        use_renderer=render,
        use_np_model=False,
        toric=False,
        dummy_simulation=False,
        debug=True,
        seed=0,
    )
    env_mesa = BaseGridRogRLEnv(config=env_config)

    render = "simple"  # "ansi"  # change to "PIL"
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
        use_np_model=False,
        toric=False,
        dummy_simulation=False,
        debug=True,
        seed=0,
    )
    free_exploration_env_mesa = FreeExplorationEnv(config=env_config)
    return [env_mesa, free_exploration_env_mesa]


@pytest.fixture(scope="module")
def all_envs():
    render = "simple"  # "ansi"  # change to "PIL"
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
            "latent_period_mu": 2 * 4,
            "latent_period_sigma": 0,
            "incubation_period_mu": 5 * 4,
            "incubation_period_sigma": 0,
            "recovery_period_mu": 14 * 4,
            "recovery_period_sigma": 0,
        },
        max_simulation_timesteps=200,
        early_stopping_patience=14,
        use_renderer=render,
        use_np_model=True,
        toric=False,
        dummy_simulation=False,
        debug=True,
        seed=0,
    )
    env = BaseGridRogRLEnv(config=env_config)

    render = "simple"  # "ansi"  # change to "PIL"
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
        use_np_model=True,
        toric=False,
        dummy_simulation=False,
        debug=True,
        seed=0,
    )
    free_exploration_env = FreeExplorationEnv(config=env_config)
    return [env, free_exploration_env]


@pytest.fixture(scope="module")
def env():
    render = "simple"  # "ansi"  # change to "PIL"
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
            "latent_period_mu": 2 * 4,
            "latent_period_sigma": 0,
            "incubation_period_mu": 5 * 4,
            "incubation_period_sigma": 0,
            "recovery_period_mu": 14 * 4,
            "recovery_period_sigma": 0,
        },
        max_simulation_timesteps=200,
        early_stopping_patience=14,
        use_renderer=render,
        use_np_model=True,
        toric=False,
        dummy_simulation=False,
        debug=True,
        seed=0,
    )
    env = BaseGridRogRLEnv(config=env_config)
    return env


@pytest.fixture(scope="module")
def free_exploration_env():
    render = "simple"  # "ansi"  # change to "PIL"
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
        use_np_model=True,
        toric=False,
        dummy_simulation=False,
        debug=True,
        seed=0,
    )
    free_exploration_env = FreeExplorationEnv(config=env_config)
    return free_exploration_env


@pytest.fixture(scope="module")
def names():
    return ["BaseGridRogRLEnv-v0", "FreeExplorationEnv-v0"]
