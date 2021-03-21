"""Top-level package for Rog RL."""

__author__ = """Sharada Mohanty"""
__email__ = "spmohanty91@gmail.com"
__version__ = "0.1.0"

from rog_rl.envs.base_grid_rog_rl_env import BaseGridRogRLEnv  # noqa
from rog_rl.envs.free_exploration_env import FreeExplorationEnv  # noqa
from rog_rl.envs.fixed_order_exploration_env import FixedOrderExplorationEnv  # noqa

from gym.envs.registration import register

register(
    id="RogRLEnv-v0",
    entry_point="rog_rl.env:RogRLEnv",
)
