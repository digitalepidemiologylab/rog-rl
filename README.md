# Rog RL

[![Build Status](https://travis-ci.org/spMohanty/RogRL.svg?branch=master)](https://travis-ci.org/spMohanty/RogRL)
[![codecov](https://codecov.io/gh/spMohanty/RogRL/branch/master/graph/badge.svg)](https://codecov.io/gh/spMohanty/RogRL)
[![Documentation Status](https://readthedocs.org/projects/rogrl/badge/?version=latest)](https://rogrl.readthedocs.io/en/latest/?badge=latest)

![](https://i.imgur.com/qPAu80s.png)

A simple Gym environment for RL experiments around disease transmission in a grid world environment.

## Installation

```bash
pip install -U git+git://gitlab.aicrowd.com/rog-rl/rog-rl.git
rog-rl-demo
```

and if everything went well, ideally you should see something along the lines of ![this](https://imgur.com/yZVvaDq.png).

## Usage

```python
#! /usr/bin/env python

from rog_rl import RogSimEnv

env = RogSimEnv()

observation = env.reset()
done = False
while not done:
    observation, reward, done, info = env.step(env.action_space.sample())
```

### Usage with Simple Renderer

```python

from rog_rl import RogSimEnv
render = "simple"
env_config = dict(
                width=10,
                height=10,
                population_density=0.80,
                vaccine_density=1.0,
                initial_infection_fraction=0.02,
                use_renderer=render,
                debug=True)

env = RogSimEnv(env_config)

observation = env.reset()
done = False
env.render(mode=render)
while not done:
    _action = input("Enter action - ex: [1, 4, 2] : ")
    if _action.strip() == "":
        _action = env.action_space.sample()
    else:
        _action = [int(x) for x in _action.split()]
        assert _action[0] in [0, 1]
        assert _action[1] in list(range(env._model.width))
        assert _action[2] in list(range(env._model.height))
    print("Action : ", _action)
    observation, reward, done, info = env.step(_action)
    env.render(mode=render)
```

## Available Environment Flavours

We provide multiple sets of similar grid based environment with varying degree of sophistication

### Rog-RL Single Agent

```python
    from rog_rl import RogSimSingleAgentEnv
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
            "incubation_period_sigma": 0,
            "recovery_period_mu": 20,
            "recovery_period_sigma": 0,
        },
        vaccine_score_weight=0.5,
        max_simulation_timesteps=20 * 20 * 10,
        early_stopping_patience=20,
        use_renderer=render,  # can be "human", "ansi"
        use_np_model=True,
        toric=False,
        dummy_simulation=False,
        debug=True,
        seed=0)
    env = RogSimSingleAgentEnv(config=env_config)
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
        print("""
        Valid Actions :
            MOVE_N = 0
            MOVE_E = 1
            MOVE_W = 2
            MOVE_S = 3

            VACCINATE = 4
            SIM_TICK = 5
        """)
        env.action_space.seed(k)
        _action = env.action_space.sample()

        print("Action : ", _action)
        observation, reward, done, info = env.step(_action)

        if not record:
            env.render(mode=render)
        print("Vacc_agent_location : ", env.vacc_agent_x, env.vacc_agent_y)
        k += 1
        print("="*100)
```

### Rog-RL Single Agent Action

```python
    from rog_rl import RogSimSingleAgentActionEnv
    render = "simple"  # "ansi"
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
            "incubation_period_sigma": 0,
            "recovery_period_mu": 20,
            "recovery_period_sigma": 0,
        },
        vaccine_score_weight=0.5,
        max_simulation_timesteps=20 * 20 * 10,
        early_stopping_patience=20,
        use_renderer=render,
        use_np_model=True,
        toric=False,
        dummy_simulation=False,
        debug=True,
        seed=0)
    env = RogSimSingleAgentActionEnv(config=env_config)
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
        print("""
        Valid Actions :
            MOVE = 0
            VACCINATE = 1
        """)
        env.action_space.seed(k)
        _action = env.action_space.sample()

        print("Action : ", _action)
        observation, reward, done, info = env.step(_action)

        if not record:
            env.render(mode=render)
        print("Vacc_agent_location : ", env.vacc_agent_x, env.vacc_agent_y)
        k += 1
        print("="*100)
```

### Rog-RL State

Wrapper around an existing rog sim environment specified by adding the name argument as follows

```console
env = RogSimState(config=env_config, name="RogRLSingleAgent-v0")
```

The wrapper provides additional methods for saving and reverting to states as shown below

```python
    render = "ansi"  # change to "simple"
    env_config = dict(
        width=4,
        height=4,
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
        max_simulation_timesteps=20 * 20 * 10,
        early_stopping_patience=20,
        use_renderer=render,
        use_np_model=True,
        toric=False,
        dummy_simulation=False,
        debug=True,
        seed=0)
    env = RogSimState(config=env_config, name="RogRLSingleAgent-v0")
    print("USE RENDERER ?", env.env.use_renderer)
    record = False
    if record:
        # records the the rendering in the `recording` folder
        env = wrappers.Monitor(env, "recording", force=True)
    observation = env.reset()
    done = False
    k = 0
    states = None
    if not record:
        env.render(mode=render)
    while not done:
        env.action_space.seed(k)
        _action = env.action_space.sample()
        print("Action : ", _action)
        observation, reward, done, info = env.step(_action)

        if not record:
            env.render(mode=render)
        k += 1
        print("="*100)
        if k == 3:
            # save state
            states = env.get_state()
        if k == 6:
            # reset to saved state
            env.set_state(states)
```

## Available Configurations

You can instantiate a RogSim environment with the following configuration options

```python


_config =  dict(
    width=50, # width of the grid
    height=50, # height of the grid
    population_density=0.75, # %-age of the grid to be filled by agents
    vaccine_density=0.05, # no. of vaccines available as a fractions of the population
    initial_infection_fraction=0.1, # %-age of agents which are infected in the beginning
    initial_vaccination_fraction=0.05,# %-age of agents which are vaccinated in the beginning
    prob_infection=0.2, # probability of infection transmission on a single contact
    prob_agent_movement=0.0, # probability that an agent will attempt to move an empty cell around it
    disease_planner_config={
        "latent_period_mu" :  2 * 4,
        "latent_period_sigma" :  0,
        "incubation_period_mu" :  5 * 4,
        "incubation_period_sigma" :  0,
        "recovery_period_mu" :  14 * 4,
        "recovery_period_sigma" :  0,
    },
    max_timesteps=200, # maximum timesteps per episode
    early_stopping_patience=14, # in-simulator steps to wait with the same susceptible population fraction before concluding that the simulation has ended
    use_renderer=False, # Takes : False, "simple", "ansi" , "PIL", "human"
    toric=True, # Make the grid world toric
    dummy_simulation=False, # Send dummy observations, rewards etc. Useful when doing integration testing with RL Experiments codebase
    fast_complete_simulation=True,
    fast_forward=False, # If True, when env steps through time or ticks, the env fast forwards and runs simulation to completion
    debug=True)

env = RogEnv(config=_config)


```

## Contributing

### Writing code

When you're done making changes, check that your changes pass flake8 and the
tests::

```console
flake8 rog-rl tests
pytest --cov rog_rl
```

**To run with xvfb**

This is only for linux systems. xvfb allows to run an application with a GUI headlessly on a server, WSL etc

First Install xfvb using `sudo apt-get install -y xvfb`

```console
xvfb-run pytest --cov rog_rl
```

### Developer tips for VS Code users

Add the below items in the `settings.json` file in the location `.vscode`

```json
{
    "python.linting.flake8Enabled": true,
    "python.linting.enabled": true,
    "markdown.previewFrontMatter": "show",
    "python.formatting.provider": "autopep8",
    "editor.formatOnSave": true
}
```

It can also be useful to enable source code debugging by making the below changes in the `launch.json` file in the location `.vscode`

```json
"justMyCode": false
```

* Free software: GNU General Public License v3
* Documentation: https://rog-rl.aicrowd.com/.

## Author

* Sharada Mohanty
