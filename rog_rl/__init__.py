"""Top-level package for Rog RL."""

__author__ = """Sharada Mohanty"""
__email__ = 'spmohanty91@gmail.com'
__version__ = '0.1.0'

from rog_rl.envs.rog_sim_env import RogSimEnv  # noqa
from rog_rl.envs.rog_sim_single_agent_env import RogSimSingleAgentEnv  # noqa
from rog_rl.envs.rog_sim_single_agent_action_env import RogSimSingleAgentActionEnv  # noqa

from gym.envs.registration import register

register(id='RogRLBase-v0',
         entry_point='rog_rl.env:RogSimBaseEnv',
         )
