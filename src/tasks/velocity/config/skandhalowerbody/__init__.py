from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  skandha_lower_body_flat_env_cfg,
  skandha_lower_body_rough_env_cfg,
)
from .rl_cfg import skandha_lower_body_ppo_runner_cfg

register_mjlab_task(
  task_id="SkandhaLowerBody-Rough",
  env_cfg=skandha_lower_body_rough_env_cfg(),
  play_env_cfg=skandha_lower_body_rough_env_cfg(play=True),
  rl_cfg=skandha_lower_body_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="SkandhaLowerBody-Flat",
  env_cfg=skandha_lower_body_flat_env_cfg(),
  play_env_cfg=skandha_lower_body_flat_env_cfg(play=True),
  rl_cfg=skandha_lower_body_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
