import gym

gym.envs.register(
     id='RogRLSingleAgent-v0',
     entry_point='rog_rl.envs.rog_sim_single_agent_env:RogSimSingleAgentEnv'
)


gym.envs.register(
     id='RogRL-v0',
     entry_point='rog_rl.envs.rog_sim_env:RogSimEnv'
)

gym.envs.register(
     id='RogRLSingleAgentAction-v0',
     entry_point='rog_rl.envs.rog_sim_single_agent_env:RogSimSingleAgentActionEnv'
)