"""SkandhaLowerBody constants.

SkandhaLowerBody is a 12-DOF lower-body-only derivative of ``skandharobot``
(see ``src/assets/robots/skandharobot/skandha_robot_constants.py``), matching the
real "hip down" test-rig hardware in the standalone lower-body URDF
(``lower_body.urdf.xacro``): that description mounts ``hip_link`` directly on a
fixed bench gantry (a physical test stand, not simulated here) with **no torso
body and no waist joint above it at all** -- the waist joint only exists as the
torso<->pelvis connection in the full humanoid, so it has no meaning once the
torso is removed.

Earlier revisions of this file mistakenly kept ``base_link`` (the torso) as the
floating base with ``hip_link`` hanging below it through ``waist_joint`` -- and
since that joint had no actuator (see the prior commit that commented out
``SKANDHA_LOWER_BODY_ACTUATOR_WAIST``), it was left in the compiled model as a
**free-swinging, undamped, unactuated hinge** between the (nonexistent-in-reality)
torso and the pelvis. This has been corrected: ``hip_link`` is now the floating
base directly (its own free joint), matching the real hardware exactly:
  - ``base_link`` and ``waist_joint`` are removed entirely -- not merely
    unactuated, gone from the kinematic tree.
  - Both 6-DOF legs (hip roll/yaw/pitch, knee pitch, ankle pitch/roll) are
    unchanged, still hanging off ``hip_link`` with identical transforms.
  - The hip-mounted IMU bracket bodies are unchanged, now direct children of the
    root ``hip_link`` instead of grandchildren through ``base_link``.

Every joint range, effort limit, and PD gain below is copied verbatim from
``skandha_robot_constants.py`` for the joints that survive -- this is the same
physical leg hardware, just fewer actuated joints, not a re-derivation.

## Total mass

Compiled model total mass: **~35.9 kg** (recomputed after removing ``base_link``;
the prior, incorrect torso-inclusive figure was ~45.14 kg).

## Home keyframe

Reuses the same leg joint angles as full Skandha's home keyframe (leg kinematics
are unaffected by which body is the floating root) with the root height adjusted
for the new root body: ``hip_link``'s own world Z is full Skandha's base height
(0.8692443230706399) plus the (negative) local offset ``hip_link`` used to sit at
under ``base_link`` (-0.0316317206795731), giving 0.8376126023910668. Verified
directly on the corrected model: all 8 foot-contact-sphere geoms land within
<1 mm of world Z = 0.

## Actuators

Unchanged from ``skandha_robot_constants.py`` for every joint group that
survives: hip_roll, hip_yaw, hip_pitch, knee_pitch, ankle_pitch, ankle_roll.
There is no waist actuator group -- there is no waist joint.
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

SKANDHA_LOWER_BODY_XML: Path = (
  SRC_PATH
  / "assets"
  / "robots"
  / "skandhalowerbody"
  / "xmls"
  / "SkandhaLowerBody.xml"
)
assert SKANDHA_LOWER_BODY_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, SKANDHA_LOWER_BODY_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(SKANDHA_LOWER_BODY_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
#
# Gains/limits copied verbatim from skandha_robot_constants.py -- same physical
# leg/waist hardware, just a subset of the full robot's actuated joints.
##

SKANDHA_LOWER_BODY_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(r"hip_roll_[rl]_joint",),
  stiffness=180.0,
  damping=6.0,
  effort_limit=235.0,
  armature=0.01,
)
SKANDHA_LOWER_BODY_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(r"hip_yaw_[rl]_joint",),
  stiffness=180.0,
  damping=6.0,
  effort_limit=235.0,
  armature=0.01,
)
SKANDHA_LOWER_BODY_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(r"hip_pitch_[rl]_joint",),
  stiffness=220.0,
  damping=8.0,
  effort_limit=330.0,
  armature=0.01,
)
SKANDHA_LOWER_BODY_ACTUATOR_KNEE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(r"knee_pitch_[rl]_joint",),
  stiffness=250.0,
  damping=8.0,
  effort_limit=330.0,
  armature=0.01,
)
SKANDHA_LOWER_BODY_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(r"ankle_pitch_[rl]_joint",),
  stiffness=60.0,
  damping=2.5,
  effort_limit=55.0,
  armature=0.01,
)
SKANDHA_LOWER_BODY_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(r"ankle_roll_[rl]_joint",),
  stiffness=60.0,
  damping=2.5,
  effort_limit=55.0,
  armature=0.01,
)


##
# Keyframe config.
##

# See module docstring: identical leg joint values to full Skandha's home
# keyframe, root height adjusted for hip_link now being the floating root
# directly. Verified flat-footed (<1mm) on the corrected model.
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.8376126023910668),
  joint_pos={
    "hip_pitch_r_joint": 0.20573850000000002,
    "hip_pitch_l_joint": -0.20573850000000002,
    "knee_pitch_r_joint": -0.18223825,
    "knee_pitch_l_joint": 0.18223825,
    "ankle_pitch_r_joint": -0.08492277143060875,
    "ankle_pitch_l_joint": 0.0849033936169159,
  },
  joint_vel={".*": 0.0},
)


##
# Collision config.
##

# Same convention as skandharobot: crank/motor stub bodies excluded from
# collision, feet use explicit contact-sphere geoms.
_COLLISION_GEOM_EXPR = (
  r"^(?!.*crank).*_collision$",
  r".*_contact\d+$",
)

FULL_COLLISION = CollisionCfg(
  geom_names_expr=_COLLISION_GEOM_EXPR,
  condim={r".*_contact\d+$": 3, r".*_collision$": 1},
  priority={r".*_contact\d+$": 1},
  friction={r".*_contact\d+$": (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=_COLLISION_GEOM_EXPR,
  contype=0,
  conaffinity=1,
  condim={r".*_contact\d+$": 3, r".*_collision$": 1},
  priority={r".*_contact\d+$": 1},
  friction={r".*_contact\d+$": (0.6,)},
)

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r".*_contact\d+$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

SKANDHA_LOWER_BODY_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    SKANDHA_LOWER_BODY_ACTUATOR_HIP_ROLL,
    SKANDHA_LOWER_BODY_ACTUATOR_HIP_YAW,
    SKANDHA_LOWER_BODY_ACTUATOR_HIP_PITCH,
    SKANDHA_LOWER_BODY_ACTUATOR_KNEE_PITCH,
    SKANDHA_LOWER_BODY_ACTUATOR_ANKLE_PITCH,
    SKANDHA_LOWER_BODY_ACTUATOR_ANKLE_ROLL,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_skandha_lower_body_cfg() -> EntityCfg:
  """Get a fresh SkandhaLowerBody configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when the
  config is shared across multiple places.
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=SKANDHA_LOWER_BODY_ARTICULATION,
  )


SKANDHA_LOWER_BODY_ACTION_SCALE: dict[str, float] = {}
for a in SKANDHA_LOWER_BODY_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    SKANDHA_LOWER_BODY_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_skandha_lower_body_cfg())

  viewer.launch(robot.spec.compile())
