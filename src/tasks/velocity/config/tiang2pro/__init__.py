from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
    tiangong_flat_env_cfg,
    tiangong_rough_env_cfg,
)
from .rl_cfg import tiangong_ppo_runner_cfg

register_mjlab_task(
    task_id="Tiangong-Rough",
    env_cfg=tiangong_rough_env_cfg(),
    play_env_cfg=tiangong_rough_env_cfg(play=True),
    rl_cfg=tiangong_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Tiangong-Flat",
    env_cfg=tiangong_flat_env_cfg(),
    play_env_cfg=tiangong_flat_env_cfg(play=True),
    rl_cfg=tiangong_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)