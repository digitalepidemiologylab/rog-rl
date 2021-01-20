import pytest
from rog_rl import RogSimEnv  # noqa
from rog_rl import RogSimSingleAgentEnv  # noqa
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
            "latent_period_mu":  2 * 4,
            "latent_period_sigma":  0,
            "incubation_period_mu":  5 * 4,
            "incubation_period_sigma":  0,
            "recovery_period_mu":  14 * 4,
            "recovery_period_sigma":  0,
        },
        max_simulation_timesteps=200,
        early_stopping_patience=14,
        use_renderer=render,
        use_np_model=False,
        toric=False,
        dummy_simulation=False,
        debug=True,
        seed=0)
    env_mesa = RogSimEnv(config=env_config)

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
        seed=0)
    single_agent_env_mesa = RogSimSingleAgentEnv(config=env_config)
    return [env_mesa, single_agent_env_mesa]


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
            "latent_period_mu":  2 * 4,
            "latent_period_sigma":  0,
            "incubation_period_mu":  5 * 4,
            "incubation_period_sigma":  0,
            "recovery_period_mu":  14 * 4,
            "recovery_period_sigma":  0,
        },
        max_simulation_timesteps=200,
        early_stopping_patience=14,
        use_renderer=render,
        use_np_model=True,
        toric=False,
        dummy_simulation=False,
        debug=True,
        seed=0)
    env = RogSimEnv(config=env_config)

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
        seed=0)
    single_agent_env = RogSimSingleAgentEnv(config=env_config)
    return [env, single_agent_env]


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
            "latent_period_mu":  2 * 4,
            "latent_period_sigma":  0,
            "incubation_period_mu":  5 * 4,
            "incubation_period_sigma":  0,
            "recovery_period_mu":  14 * 4,
            "recovery_period_sigma":  0,
        },
        max_simulation_timesteps=200,
        early_stopping_patience=14,
        use_renderer=render,
        use_np_model=True,
        toric=False,
        dummy_simulation=False,
        debug=True,
        seed=0)
    env = RogSimEnv(config=env_config)
    return env


@pytest.fixture(scope="module")
def single_agent_env():
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
        seed=0)
    single_agent_env = RogSimSingleAgentEnv(config=env_config)
    return single_agent_env


@pytest.fixture(scope="module")
def names():
    return ['RogRL-v0', 'RogRLSingleAgent-v0']
