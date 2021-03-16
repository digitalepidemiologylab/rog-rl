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

We show the perfect ring vaccination policy trained using an RL Agent trained after 100100 envrionment steps. This agent is able to prevent vacccine spread but also ensures no wastage of vaccines in a `toric 5*5 grid*`.

![](https://i.imgur.com/qnJ1zZH.png)

```python
"normalized_vaccine_wastage": 0, "normalized_susceptible": 1.00,    "normalized_protected": 1.00
```

Notice for the toric grid, the infectious agent at cell `(4,1)` is surrounded by vaccines and hence no infection spreads and there is also no wastage of vaccines as only the 8 cells adjacent to the infectious cell is vaccinated.

## Comparing different vaccination strategy

We can compare the performance across different vaccination strategy by looking at the `normalized_susceptible`  metric.  For e.g. looking at the 2 cases below

Agent 1:

```python
"normalized_vaccine_wastage": 0.38, "normalized_susceptible": 0.63,    "normalized_protected": 1.00
```

Agent 2

```python
"normalized_vaccine_wastage": 0.32, "normalized_susceptible": 0.69,    "normalized_protected": 1.00
```

We can see Agent 2 is better as it has a higher `normalized_susceptible` score. It is also easy to see this by just comparing the `normalized_vaccine_wastage` score as the `normalized_protected = 1` for both of them. If the `normalized_protected` score was higher for Agent 1, then it would not be obvious and we can then use the `normalized_susceptible` score to decide on the better agent.

## Is ring vaccination always the best strategy?

We can have cases where ring vaccination is not always the best strategy. One such case happens when the number of vaccinations available is not enough to ring vaccinate.

Lets assume we have a non-toric grid with width,height as 7,5 respectively with total number of agents as 35. There are 2 Infectious agents and we have only 8 vaccines. We also for simplicity assume that the `simulation_single_tick = True`, so the vaccinations happen in the first time step and then the simulation is allowed to run till completion.

![](https://i.imgur.com/VQa47d8.png)

We can see that the number of vaccines required to fully ring vaccinate the 2 infectious agents is 13. If we follow a simple ring vaccination strategy with the available 8 vaccines, we can only ring vaccinate one infectious agent and the infection spreads due to the other agent. If we run the simulation, we end up with the following scenario as shown in the screenshot below for the last 2 steps.

![](https://i.imgur.com/P26glb2.png)

The normalised metrics are as follows

```python
'normalized_vaccine_wastage': -0.25, 'normalized_susceptible': 0.0, 'normalized_protected': 0.24242424242424243
```

Now, if we were to intelligently try to create grid areas seperating different infectious agents. The last 2 steps now becomes

![](https://i.imgur.com/6XaS3Ci.png)


```python
'normalized_vaccine_wastage': -0.3, 'normalized_susceptible': 1.0, 'normalized_protected': 0.8181818181818181
```


If we make the above problem further difficult, by providing only 5 vaccines, then the intelligent vaccination approach is still able to protect many agents, but the ring vaccination performs worse.

Below screenshot shows the last 2 steps for the ring vaccination strategy

![](https://i.imgur.com/5BzRdlj.png)


```python
'normalized_vaccine_wastage': -0.4, 'normalized_susceptible': 0.0, 'normalized_protected': 0.15151515151515152

```

Similarly for a grid vaccination approach, the last 2 steps are

![](https://i.imgur.com/NGuR2Bu.png)


```python
'normalized_vaccine_wastage': -0.4, 'normalized_susceptible': 1.0, 'normalized_protected': 0.7575757575757575
```


We can clearly see that the grid vaccination performs far better than the ring vaccination. It is able to protect more susceptible agents and also conserve vaccines.

This is only a simple scenario involving 2 infection clusters in a simple `7*5`  grid environment and hence it was easy to find that a grid based vaccination strategy is better than a ring vaccination strategy. When the environment becomes much larger with different infection clusters (which is also what happens practically in real life), it becomes difficult to find these grids and vaccinate.

However, we think an intelligent RL (Reinforcement Learning) agent will be able to learn and provide us with optimal solutions than can help guide future vaccination policy and strategy. We have been able  to train a RL agent that was able to learn ring vaccination by providing a local view of an agent with various local radius sizes ranging from the nearest neighbours to a view involving agents seperated by upto 3 cells. These successful experiments provides us with the impetus so that we can in future also successfully train RL agents that can learn various other vaccination strategies in large environments. A RL solution will also have the added benefit of being scalable which is a critical requirement.

## Apply stochasticity to the problem

We notice that our parameters like the`initial_infection_fraction`, `vaccine density`, `prob_infection` are all constant, but in a real life scenario these are uncertain estimates. We can add a stochastic element to these data and that further increases the complexity of the problem.

We think this also makes an RL solution more well suited to a hand designed expert solution.

## Incorporating Vaccine efficiency

Currently we also assume that the vaccination has a `100%` success rate where an vaccinated agent can not be infected. However as typically seen in practice, vaccines have an efficacy rate less than `100%` thereby making the problem of vaccination tricky.

## Adding multiple time steps for vaccine intravention

We have in all previous examples assumed that we only make the vaccination decision once, and then the simulation happens till the terminal condition is reached. If we allow for vaccine intraventions as the simulation progresses through different time steps, the problem gets more complicated where decisions additionally rely on whether to vaccinate ahead of time or to wait and watch and then vaccinate.
