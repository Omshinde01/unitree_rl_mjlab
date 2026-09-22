from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  skandha_robot_flat_env_cfg,
  skandha_robot_rough_env_cfg,
)
from .rl_cfg import skandha_robot_ppo_runner_cfg

register_mjlab_task(
  task_id="Skandha-Rough",
  env_cfg=skandha_robot_rough_env_cfg(),
  play_env_cfg=skandha_robot_rough_env_cfg(play=True),
  rl_cfg=skandha_robot_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Skandha-Flat",
  env_cfg=skandha_robot_flat_env_cfg(),
  play_env_cfg=skandha_robot_flat_env_cfg(play=True),
  rl_cfg=skandha_robot_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
