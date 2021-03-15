# Fixed Order Exploration Environment

We explain the working of a fixed order environment using a simple example of 5*5 grid. We use the ANSI Renderer to show the different examples. The configuration used for the sample environment is as follows:

```python
    np.random.seed(100)
    render = "ansi"  # "ansi"  # change to "human"
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
        simulation_single_tick=True,
        debug=True,
        seed=0)
```


The fixed order environment is a simple environment where the agent starts from the top left corner and moves in a fixed order to the the botton right. We show the first 3 steps below where the agent moves from location `(0,0)` to `(2,0)` 

![](https://i.imgur.com/qljhCfI.png)


Looking at the various agent states, we can see that the grid has `1` infectious agent and remaining `24` are susceptible agents. Our objective is to stop the infection from spreading to the remaining 24 susceptible agent using the least number of vaccines.

If agent vacciantes the cell positions `(0,0) , (2,0) , (1,0),  (1,1) and (2,1)` if the `toric` is `False` or no horizonal or vertical wrapping of environment is allowed.

If `toric` is `True` , then the agent has to vaccinate 3 other additional cell positions `(4,0), (4,1) and (4,2)`

This vaccination policy , also known as `ring vaccination`  vaccinates the agents surrounding the infectious agents thereby preventing any spread of infection.

The ANSI renderer also shows us other useful information like the

* % of agents in each of the states - Susceptible, Infectious, Recovered and Vaccinated

* Number of Simulation Ticks or time steps and the Environment Steps
* Vaccine Budget or vaccines remaining

Now moving ahead to envrionment step 5, we can see at step 4, the agent chooses to vaccinates cell `0,3` , we can see that our vaccine budget reduces by `1` and we have an agent in vaccinated cell. Note that vaccination is only successful if we apply it on the `susceptible agent`, and only then the agent state changes to `vaccinated`.

![](https://i.imgur.com/iYIHpyg.png)


Moving on, the agent moving in a fixed order can cover the entire grid after 25 steps. The `single_simulation_tick` is set to `True`  which means the agent action is freezed and the simulation will move towards its terminal state after a simulation tick occurs. This happens automatically for the fixed order environment once the entire grid is covered by the agent.

![](https://i.imgur.com/gDtjImc.png)

Notice in the 25th step , our random policy has vaccinated `52%` of the agents or `52% * 25 = 13` agents. We can also see that the agents vaccinated are quite far from ideal as in the last step when the simulation completes, many susceptible agents were infected and eventually moved to the recovered state. The vaccines were also wasted on many agents who were far from the infectious agent.

The visualisation helps us understand how effective our policy is working.  We have also desinged normalised metrics to help us understand how `good` or `bad` our policy is against our twin objectives of `Stopping Infection Spread` and `Minimise utilisation of vaccines`.

As explained in the Environment metrics, we designed 3 metrics for this purpose with the 2 metrics for checking against the 2 objectives and the 3rd metric combining the 2 objectives into a single number.

We show the metric values for this random policy below

```python
'normalized_vaccine_wastage': 0.47, 'normalized_susceptible': 0.0, 'normalized_protected': 0.54
```

Good Policy

We show a better policy trained using an RL Agent trained after 35000 envrionment steps, which is able to prevent vacccine spread but does cause some wastage of vaccines in a `toric 5*5 grid*`. We show the last 2 steps for brevity below


![](https://i.imgur.com/tWlTjk0.png)

We can see that the initial infectious agent moves to recovered, but no other agent is infected or has eventually moved to recovered, thereby showing that our vaccination has prevented other agents from getting infected. Howver, we can see that there are some wasted vaccines. Hence, this is not the perfect vaccination strategy.

We show the metric values for this RL policy below

```python
"normalized_vaccine_wastage": 0.38, "normalized_susceptible": 0.63,    "normalized_protected": 1.00
```

Best Policy
