"""KodyRobot constants.

Actuator configuration is intentionally defined per physical actuated joint so that
each actuator has its own stiffness, damping, effort limit, and armature.

The *_support joints in the KodyRobot MJCF are treated as passive mechanical
support joints and are therefore not included in the position-actuator list.
"""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg


##
# MJCF and assets.
##

KODY_ROBOT_XML: Path = (
  SRC_PATH / "assets" / "robots" / "kody_robot" / "xmls" / "KodyRobot.xml"
)
assert KODY_ROBOT_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, KODY_ROBOT_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(KODY_ROBOT_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
#
# Each physical actuator is kept separate rather than grouped with regexes.
# This makes joint-level tuning straightforward.
##

# ---------------------------------------------------------------------------
# Right leg
# ---------------------------------------------------------------------------

KODY_ACTUATOR_RIGHT_HIP_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=("hip_roll_r_joint",),
  stiffness=180.0,
  damping=6.0,
  effort_limit=235.0,
  armature=0.01,
)

KODY_ACTUATOR_RIGHT_HIP_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=("hip_yaw_r_joint",),
  stiffness=180.0,
  damping=6.0,
  effort_limit=235.0,
  armature=0.01,
)

KODY_ACTUATOR_RIGHT_HIP_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=("hip_pitch_r_joint",),
  stiffness=220.0,
  damping=8.0,
  effort_limit=330.0,
  armature=0.01,
)

KODY_ACTUATOR_RIGHT_KNEE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=("knee_pitch_r_joint",),
  stiffness=250.0,
  damping=8.0,
  effort_limit=330.0,
  armature=0.01,
)

KODY_ACTUATOR_RIGHT_ANKLE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=("ankle_pitch_r_joint",),
  stiffness=60.0,
  damping=2.5,
  effort_limit=55.0,
  armature=0.01,
)

KODY_ACTUATOR_RIGHT_ANKLE_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=("ankle_roll_r_joint",),
  stiffness=60.0,
  damping=2.5,
  effort_limit=55.0,
  armature=0.01,
)


# ---------------------------------------------------------------------------
# Left leg
# ---------------------------------------------------------------------------

KODY_ACTUATOR_LEFT_HIP_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=("hip_roll_l_joint",),
  stiffness=180.0,
  damping=6.0,
  effort_limit=235.0,
  armature=0.01,
)

KODY_ACTUATOR_LEFT_HIP_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=("hip_yaw_l_joint",),
  stiffness=180.0,
  damping=6.0,
  effort_limit=235.0,
  armature=0.01,
)

KODY_ACTUATOR_LEFT_HIP_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=("hip_pitch_l_joint",),
  stiffness=220.0,
  damping=8.0,
  effort_limit=330.0,
  armature=0.01,
)

KODY_ACTUATOR_LEFT_KNEE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=("knee_pitch_l_joint",),
  stiffness=250.0,
  damping=8.0,
  effort_limit=330.0,
  armature=0.01,
)

KODY_ACTUATOR_LEFT_ANKLE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=("ankle_pitch_l_joint",),
  stiffness=60.0,
  damping=2.5,
  effort_limit=55.0,
  armature=0.01,
)

KODY_ACTUATOR_LEFT_ANKLE_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=("ankle_roll_l_joint",),
  stiffness=60.0,
  damping=2.5,
  effort_limit=55.0,
  armature=0.01,
)


# ---------------------------------------------------------------------------
# Waist
# ---------------------------------------------------------------------------

KODY_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_joint",),
  stiffness=150.0,
  damping=5.0,
  effort_limit=91.0,
  armature=0.01,
)


# ---------------------------------------------------------------------------
# Right arm
# ---------------------------------------------------------------------------

KODY_ACTUATOR_RIGHT_SHOULDER_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=("shoulder_pitch_r_joint",),
  stiffness=90.0,
  damping=4.0,
  effort_limit=91.0,
  armature=0.01,
)

KODY_ACTUATOR_RIGHT_SHOULDER_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=("shoulder_roll_r_joint",),
  stiffness=85.0,
  damping=4.0,
  effort_limit=95.0,
  armature=0.01,
)

KODY_ACTUATOR_RIGHT_SHOULDER_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=("shoulder_yaw_r_joint",),
  stiffness=65.0,
  damping=3.0,
  effort_limit=35.0,
  armature=0.01,
)

KODY_ACTUATOR_RIGHT_ELBOW_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=("elbow_pitch_r_joint",),
  stiffness=60.0,
  damping=3.0,
  effort_limit=35.0,
  armature=0.01,
)

KODY_ACTUATOR_RIGHT_ELBOW_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=("elbow_yaw_r_joint",),
  stiffness=45.0,
  damping=2.5,
  effort_limit=24.0,
  armature=0.01,
)

KODY_ACTUATOR_RIGHT_WRIST_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=("wrist_pitch_r_joint",),
  stiffness=20.0,
  damping=1.0,
  effort_limit=6.3,
  armature=0.01,
)

KODY_ACTUATOR_RIGHT_WRIST_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=("wrist_roll_r_joint",),
  stiffness=20.0,
  damping=1.0,
  effort_limit=6.3,
  armature=0.01,
)


# ---------------------------------------------------------------------------
# Left arm
# ---------------------------------------------------------------------------

KODY_ACTUATOR_LEFT_SHOULDER_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=("shoulder_pitch_l_joint",),
  stiffness=90.0,
  damping=4.0,
  effort_limit=91.0,
  armature=0.01,
)

KODY_ACTUATOR_LEFT_SHOULDER_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=("shoulder_roll_l_joint",),
  stiffness=85.0,
  damping=4.0,
  effort_limit=95.0,
  armature=0.01,
)

KODY_ACTUATOR_LEFT_SHOULDER_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=("shoulder_yaw_l_joint",),
  stiffness=65.0,
  damping=3.0,
  effort_limit=35.0,
  armature=0.01,
)

KODY_ACTUATOR_LEFT_ELBOW_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=("elbow_pitch_l_joint",),
  stiffness=60.0,
  damping=3.0,
  effort_limit=35.0,
  armature=0.01,
)

KODY_ACTUATOR_LEFT_ELBOW_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=("elbow_yaw_l_joint",),
  stiffness=45.0,
  damping=2.5,
  effort_limit=24.0,
  armature=0.01,
)

KODY_ACTUATOR_LEFT_WRIST_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=("wrist_roll_l_joint",),
  stiffness=20.0,
  damping=1.0,
  effort_limit=6.3,
  armature=0.01,
)

KODY_ACTUATOR_LEFT_WRIST_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=("wrist_pitch_l_joint",),
  stiffness=20.0,
  damping=1.0,
  effort_limit=6.3,
  armature=0.01,
)


# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------

KODY_ACTUATOR_HEAD_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=("head_yaw_joint",),
  stiffness=20.0,
  damping=1.0,
  effort_limit=6.3,
  armature=0.01,
)

KODY_ACTUATOR_HEAD_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=("head_pitch_joint",),
  stiffness=20.0,
  damping=1.0,
  effort_limit=6.3,
  armature=0.01,
)

KODY_ACTUATOR_HEAD_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=("head_roll_joint",),
  stiffness=15.0,
  damping=0.8,
  effort_limit=6.3,
  armature=0.01,
)


##
# Keyframe config.
##

# Zero joint pose is used deliberately as the neutral starting configuration.
# The URDF zero-pose places the feet at approximately the ground-contact height.
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.90),
  joint_pos={
    ".*": 0.0,
  },
  joint_vel={
    ".*": 0.0,
  },
)


##
# Collision config.
##

# KodyRobot does not use the H2-specific foot_1...foot_7 naming scheme.
# Therefore the generic *_collision convention is used for the converted MJCF.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={".*_collision": 1},
  friction={".*_collision": (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={".*_collision": 1},
  friction={".*_collision": (0.6,)},
)


##
# Final config.
##

KODY_ROBOT_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    # Right leg.
    KODY_ACTUATOR_RIGHT_HIP_ROLL,
    KODY_ACTUATOR_RIGHT_HIP_YAW,
    KODY_ACTUATOR_RIGHT_HIP_PITCH,
    KODY_ACTUATOR_RIGHT_KNEE_PITCH,
    KODY_ACTUATOR_RIGHT_ANKLE_PITCH,
    KODY_ACTUATOR_RIGHT_ANKLE_ROLL,

    # Left leg.
    KODY_ACTUATOR_LEFT_HIP_ROLL,
    KODY_ACTUATOR_LEFT_HIP_YAW,
    KODY_ACTUATOR_LEFT_HIP_PITCH,
    KODY_ACTUATOR_LEFT_KNEE_PITCH,
    KODY_ACTUATOR_LEFT_ANKLE_PITCH,
    KODY_ACTUATOR_LEFT_ANKLE_ROLL,

    # Waist.
    KODY_ACTUATOR_WAIST,

    # Right arm.
    KODY_ACTUATOR_RIGHT_SHOULDER_PITCH,
    KODY_ACTUATOR_RIGHT_SHOULDER_ROLL,
    KODY_ACTUATOR_RIGHT_SHOULDER_YAW,
    KODY_ACTUATOR_RIGHT_ELBOW_PITCH,
    KODY_ACTUATOR_RIGHT_ELBOW_YAW,
    KODY_ACTUATOR_RIGHT_WRIST_PITCH,
    KODY_ACTUATOR_RIGHT_WRIST_ROLL,

    # Left arm.
    KODY_ACTUATOR_LEFT_SHOULDER_PITCH,
    KODY_ACTUATOR_LEFT_SHOULDER_ROLL,
    KODY_ACTUATOR_LEFT_SHOULDER_YAW,
    KODY_ACTUATOR_LEFT_ELBOW_PITCH,
    KODY_ACTUATOR_LEFT_ELBOW_YAW,
    KODY_ACTUATOR_LEFT_WRIST_ROLL,
    KODY_ACTUATOR_LEFT_WRIST_PITCH,

    # Head.
    KODY_ACTUATOR_HEAD_YAW,
    KODY_ACTUATOR_HEAD_PITCH,
    KODY_ACTUATOR_HEAD_ROLL,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_kody_robot_cfg() -> EntityCfg:
  """Get a fresh KodyRobot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=KODY_ROBOT_ARTICULATION,
  )


KODY_ROBOT_ACTION_SCALE: dict[str, float] = {}
for a in KODY_ROBOT_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  assert s > 0.0
  for n in names:
    KODY_ROBOT_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_kody_robot_cfg())

  viewer.launch(robot.spec.compile())
