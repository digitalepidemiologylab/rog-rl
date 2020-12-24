import pytest
from rog_rl.env import ActionType
from rog_rl.envs.rog_sim_single_agent_env import ActionType as SingleAgentActionType
import numpy as np
from rog_rl.agent_state import AgentState
from collections import defaultdict
from itertools import permutations
from scipy import stats
import random
import warnings
warnings.filterwarnings("ignore")


n_runs = 100
seed = 100
np.random.seed(seed)
random.seed(seed)

def kl_divergence(p, q):
    p = np.array(p)
    q = np.array(q)
    size_p = len(p)
    size_q = len(q)
    if size_p == size_q:
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))
    else:
        min_size = min(size_p, size_q)
        if size_p > size_q:
            idx = np.random.choice(size_p, min_size, replace=False)
            p = p[idx]
        else:
            idx = np.random.choice(size_q, min_size, replace=False)
            q = q[idx]

        return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def collect_env_data(env):
    infos_runs = []
    for i in range(n_runs):
        env.seed(i)
        infos = defaultdict(list)
        env.debug = False
        observation = env.reset()
        done = False
        k = 0

        while not done:
            env.action_space.seed(k)
            _action = env.action_space.sample()
            observation, reward, done, info = env.step(_action)
            k += 1
            for _state in AgentState:
                key = "population.{}".format(_state.name)
                infos[key].append(info[key])
        infos_runs.append(infos)

    return infos_runs


@pytest.mark.skip(reason="Both the envs are different and expected to fail")
def test_env_distributions(env, single_agent_env):
    run_statistical_test(env, single_agent_env)


@pytest.mark.skip(reason="legacy mesa model is not supported")
def test_env_mesa_model_distributions(env, all_mesa_envs):
    env_mesa = all_mesa_envs[0]
    run_statistical_test(env, env_mesa)


@pytest.mark.skip(reason="legacy mesa model is not supported")
def test_single_agent_env_mesa_model_distributions(single_agent_env, all_mesa_envs):
    single_agent_env_mesa = all_mesa_envs[1]
    run_statistical_test(single_agent_env, single_agent_env_mesa)


def run_statistical_test(env1, env2):

    """
    Compares 2 environments to check if the population distributions match.
    This checks for all the agent states saved in the info object

    Args:
    -------
    env: instantiated env which is for reference
    env: the instantiated env for comparison

    Note the order is important and this is not commutative

    """

    kl_values = defaultdict(list)
    infos_runs = collect_env_data(env1)

    for _state in AgentState:
        key = "population.{}".format(_state.name)
        all_perm = permutations(range(n_runs), 2)
        for perm in list(all_perm):
            idx1 = perm[0]
            idx2 = perm[1]
            info_run1 = infos_runs[idx1]
            info_run2 = infos_runs[idx2]
            kl_val = kl_divergence(info_run1[key],info_run2[key])
            if not np.isposinf(kl_val) and not np.isneginf(kl_val):
                kl_values[key].append(kl_val)

    infos_runs_other_env = collect_env_data(env2)

    kl_values_single_agent = defaultdict(list)
    for _state in AgentState:
        key = "population.{}".format(_state.name)
        all_perm = permutations(range(n_runs), 2)
        for run in range(n_runs):
            info_run1 = infos_runs[run]
            info_run2 = infos_runs_other_env[run]
            kl_val = kl_divergence(info_run1[key],info_run2[key])
            if not np.isposinf(kl_val) and not np.isneginf(kl_val):
                kl_values_single_agent[key].append(kl_val)

    percentiles_other_env = defaultdict(list)
    for _state in AgentState:
        key = "population.{}".format(_state.name)
        kl_vals = kl_values_single_agent[key]
        kl_val_dist = kl_values[key]

        for kl_val in kl_vals:
            percentile_val = stats.percentileofscore(kl_val_dist, kl_val)
            percentiles_other_env[key].append(percentile_val/100)


    for _state in AgentState:
        key = "population.{}".format(_state.name)
        perc_vals = percentiles_other_env[key]
        if len(set(perc_vals)) > 1:
            T=stats.uniform(0,1).rvs(len(perc_vals), random_state=seed)
            statistic_ad,critical_values_ad,significance_level=stats.anderson_ksamp([T,perc_vals])
            print(statistic_ad)

            # The critical values for significance levels 25%, 10%, 5%, 2.5%, 1%,
            # 0.5%, 0.1%.

            # Check at 5% confidence level
            if statistic_ad < critical_values_ad[4]:
                print(key," is uniform based on AD test", )
            else:
                print(key," is not uniform based on AD test", )

            # Fail test at 10% confidence level
            assert statistic_ad < critical_values_ad[5]

        else:
            print("Not sufficient unique values for state:",_state.name)







if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-sv", __file__]))
