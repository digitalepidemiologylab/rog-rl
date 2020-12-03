import pytest
from rog_rl.env import ActionType
from rog_rl.envs.rog_sim_single_agent_env import ActionType as SingleAgentActionType
import numpy as np
from rog_rl.agent_state import AgentState
from collections import defaultdict
from itertools import permutations
from scipy import stats


n_runs = 20
seed = 100
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
        infos = defaultdict(list)
        env.debug = False
        observation = env.reset()
        done = False
        k = 0

        while not done:
            _action = env.action_space.sample()
            observation, reward, done, info = env.step(_action)
            for _state in AgentState:
                key = "population.{}".format(_state.name)
                infos[key].append(info[key]) 
        infos_runs.append(infos)

    return infos_runs


def test_run_env(env, single_agent_env):
    np.random.seed(seed)
    kl_values = defaultdict(list)
    infos_runs = collect_env_data(env)

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

    print(kl_values)

    infos_runs_single_agent = collect_env_data(single_agent_env)

    kl_values_single_agent = defaultdict(list)
    for _state in AgentState:
        key = "population.{}".format(_state.name)
        all_perm = permutations(range(n_runs), 2)
        for run in range(n_runs): 
            info_run1 = infos_runs[run]
            info_run2 = infos_runs_single_agent[run]
            kl_val = kl_divergence(info_run1[key],info_run2[key])
            if not np.isposinf(kl_val) and not np.isneginf(kl_val):
                kl_values_single_agent[key].append(kl_val)

    percentiles_single_agent = defaultdict(list)
    for _state in AgentState:
        key = "population.{}".format(_state.name)
        kl_vals = kl_values_single_agent[key]
        kl_val_dist = kl_values[key]

        for kl_val in kl_vals:
            percentile_val = stats.percentileofscore(kl_val_dist, kl_val)
            percentiles_single_agent[key].append(percentile_val/100)


    # TODO: Add Anderson Darling test for p-value


    


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", __file__]))