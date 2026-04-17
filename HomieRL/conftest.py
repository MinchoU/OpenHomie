"""Pytest config for HomieRL: ensure isaacgym is imported before torch.

legged_gym.utils.__init__ eagerly imports helpers.py which does
`from isaacgym import gymapi`. isaacgym's gymdeps refuses if torch was
already loaded (e.g., via pytest plugin machinery). Importing isaacgym
here at conftest load time guarantees the correct order.

Additionally, `legged_gym.utils.__init__` imports task_registry whose
imports trigger `legged_gym.envs.__init__`, which re-imports back into
`legged_gym.utils.task_registry`, creating a circular-import cycle when
any `legged_gym.utils.<sub>` module is imported directly. Pre-importing
`legged_gym.envs` at collection time breaks the cycle.
"""

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

try:
    import legged_gym.envs  # noqa: F401
except ImportError:
    pass
