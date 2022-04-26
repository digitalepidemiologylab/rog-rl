from rog_rl import FixedOrderExplorationEnv
import gym
from gym import spaces, wrappers

render = "simple"  # "ansi"  # change to "human"
env_config = dict(
    width=10,
    height=10,
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
    max_simulation_timesteps=100,
    early_stopping_patience=20,
    use_renderer=render,
    use_model_np=True,
    fast_complete_simuation=True,
    toric=False,
    dummy_simulation=False,
    debug=True,
)
env = FixedOrderExplorationEnv(config=env_config)
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
print(info)