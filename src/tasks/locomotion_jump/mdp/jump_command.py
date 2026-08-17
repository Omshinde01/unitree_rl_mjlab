"""Velocity command with an additional binary jump command."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.velocity.mdp import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)


class JumpVelocityCommand(UniformVelocityCommand):
  """Uniform velocity command extended with a binary jump command.

  Command layout:

    [linear_vel_x, linear_vel_y, angular_vel_z, jump]

  where:
    jump = 0.0 -> normal locomotion
    jump = 1.0 -> jump requested
  """

  cfg: JumpVelocityCommandCfg

  def __init__(
    self,
    cfg: JumpVelocityCommandCfg,
    env: ManagerBasedRlEnv,
  ):
    super().__init__(cfg, env)

    # One additional command dimension:
    # 0 -> normal locomotion
    # 1 -> jump
    self.jump_command = torch.zeros(
      self.num_envs,
      1,
      device=self.device,
    )

  @property
  def command(self) -> torch.Tensor:
    """Return the complete [vx, vy, yaw, jump] command."""
    return torch.cat(
      (
        self.vel_command_b,
        self.jump_command,
      ),
      dim=-1,
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """Resample velocity and jump commands.

    Velocity sampling is handled by the original
    UniformVelocityCommand implementation.
    """

    # Preserve the original velocity-command behavior.
    super()._resample_command(env_ids)

    # Sample the binary jump command.
    #
    # jump_probability = probability that an environment
    # receives jump=1 when its command is resampled.
    jump = (
      torch.rand(
        len(env_ids),
        device=self.device,
      )
      < self.cfg.jump_probability
    )

    self.jump_command[env_ids, 0] = jump.float()

  def _update_command(self) -> None:
    """Update the command while preserving velocity behavior."""

    # Preserve heading-control and other existing velocity behavior.
    super()._update_command()

    # Standing environments should not request a jump.
    #
    # This keeps the existing standing behavior from the
    # velocity task intact.
    standing_env_ids = self.is_standing_env.nonzero(
      as_tuple=False
    ).flatten()

    if standing_env_ids.numel() > 0:
      self.jump_command[standing_env_ids, 0] = 0.0


@dataclass(kw_only=True)
class JumpVelocityCommandCfg(UniformVelocityCommandCfg):
  """Configuration for velocity + binary jump command."""

  # Probability of requesting a jump whenever the command
  # is resampled.
  jump_probability: float = 0.2

  @dataclass
  class Ranges(UniformVelocityCommandCfg.Ranges):
    """Velocity command ranges.

    Inherited fields:
      lin_vel_x
      lin_vel_y
      ang_vel_z
      heading
    """

  ranges: Ranges

  def build(
    self,
    env: ManagerBasedRlEnv,
  ) -> JumpVelocityCommand:
    """Build the jump-velocity command term."""
    return JumpVelocityCommand(self, env)