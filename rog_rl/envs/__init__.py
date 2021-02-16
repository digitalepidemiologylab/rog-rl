import gym

gym.envs.register(
    id='BaseGridRogRLEnv-v0',
    entry_point='rog_rl.envs.base_grid_rog_rl_env:BaseGridRogRLEnv'
)

gym.envs.register(
    id='FreeExplorationEnv-v0',
    entry_point='rog_rl.envs.free_exploration_env:FreeExplorationEnv'
)

gym.envs.register(
    id='FixedOrderExplorationEnv-v0',
    entry_point='rog_rl.envs.fixed_order_exploration_env:FixedOrderExplorationEnv'  # noqa
)

gym.envs.register(
    id='RogRLStateEnv-v0',
    entry_point='rog_rl.envs.rog_rl_state_env:RogRLStateEnv'
)
