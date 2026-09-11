"""KodyRobot velocity environment configurations."""

from src.assets.robots import (
  KODY_ROBOT_ACTION_SCALE,
  get_kody_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

# KodyRobot's floating base is `base_link` (the torso/chest), not a pelvis --
# the pelvis-like `hip_link` hangs below it through `waist_joint`. Every
# "root body" reference below (raycast frame, IMU-derived observations via
# the XML's `imu`/`imu_ang_vel`/`imu_lin_vel` sensors, viewer target, CoM
# randomization, orientation/ang-vel rewards) therefore targets `base_link`.
_ROOT_BODY = "base_link"
_FEET_SITES = ("left_foot", "right_foot")
_FEET_CONTACT_GEOMS = tuple(
  f"{link}_contact{i}" for link in ("RL_foot_roll_link", "LL_foot_roll_link") for i in range(4)
)


def kody_robot_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create KodyRobot rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 48

  cfg.scene.entities = {"robot": get_kody_robot_cfg()}

  # Set raycast sensor frame to KodyRobot's floating base.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = _ROOT_BODY

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(RL_foot_roll_link|LL_foot_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern=_ROOT_BODY, entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern=_ROOT_BODY, entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = KODY_ROBOT_ACTION_SCALE

  cfg.viewer.body_name = _ROOT_BODY

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 0.95

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = _FEET_SITES

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = _FEET_CONTACT_GEOMS
  cfg.events["base_com"].params["asset_cfg"].body_names = (_ROOT_BODY,)

  # Rationale for std values (same philosophy as unitree_h2/unitree_g1):
  # - Knees/hip_pitch get the loosest std to allow natural leg bending during stride.
  # - Hip roll/yaw stay tighter to prevent excessive lateral sway and keep gait stable.
  # - Ankle roll is very tight for balance; ankle pitch looser for foot clearance.
  # - Waist stays tight to keep the torso/pelvis coupling stable (single-DOF here).
  # - Shoulders/elbows get moderate freedom for natural arm swing during walking.
  # - Wrists and head are loose since they barely affect balance.
  # Running values are ~1.5-2x walking values to accommodate larger motion range.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Legs.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee_pitch.*": 0.5,
    r".*ankle_pitch.*": 0.15,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist.*": 0.15,
    # Arms.
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow_pitch.*": 0.1,
    r".*elbow_yaw.*": 0.1,
    r".*wrist_roll.*": 0.1,
    r".*wrist_pitch.*": 0.1,
    # Head.
    r".*head_yaw.*": 0.2,
    r".*head_pitch.*": 0.15,
    r".*head_roll.*": 0.15,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Legs.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.25,
    r".*hip_yaw.*": 0.25,
    r".*knee_pitch.*": 0.5,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist.*": 0.25,
    # Arms.
    r".*shoulder_pitch.*": 0.25,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow_pitch.*": 0.1,
    r".*elbow_yaw.*": 0.1,
    r".*wrist_roll.*": 0.1,
    r".*wrist_pitch.*": 0.1,
    # Head.
    r".*head_yaw.*": 0.25,
    r".*head_pitch.*": 0.2,
    r".*head_roll.*": 0.2,
  }

  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = (_ROOT_BODY,)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (_ROOT_BODY,)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = _FEET_SITES
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = _FEET_SITES
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def kody_robot_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create KodyRobot flat terrain velocity configuration."""
  cfg = kody_robot_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg
