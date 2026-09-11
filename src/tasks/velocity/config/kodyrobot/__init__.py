from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  kody_robot_flat_env_cfg,
  kody_robot_rough_env_cfg,
)
from .rl_cfg import kody_robot_ppo_runner_cfg


register_mjlab_task(
  task_id="KodyRobot-Rough",
  env_cfg=kody_robot_rough_env_cfg(),
  play_env_cfg=kody_robot_rough_env_cfg(play=True),
  rl_cfg=kody_robot_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)


register_mjlab_task(
  task_id="KodyRobot-Flat",
  env_cfg=kody_robot_flat_env_cfg(),
  play_env_cfg=kody_robot_flat_env_cfg(play=True),
  rl_cfg=kody_robot_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
