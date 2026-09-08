from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
    kodyrobot_flat_env_cfg,
    kodyrobot_rough_env_cfg,
)
from .rl_cfg import kodyrobot_ppo_runner_cfg

register_mjlab_task(
    task_id="KodyRobot-Rough",
    env_cfg=kodyrobot_rough_env_cfg(),
    play_env_cfg=kodyrobot_rough_env_cfg(play=True),
    rl_cfg=kodyrobot_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="KodyRobot-Flat",
    env_cfg=kodyrobot_flat_env_cfg(),
    play_env_cfg=kodyrobot_flat_env_cfg(play=True),
    rl_cfg=kodyrobot_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)