"""SkandhaLowerBody constants.

SkandhaLowerBody is a 13-DOF lower-body-only derivative of ``skandharobot``
(see ``src/assets/robots/skandharobot/skandha_robot_constants.py``), produced by
deleting the arm, head, and chest-camera subtrees from ``SkandhaRobot.xml`` and
keeping everything else (torso, waist, both legs, hip-mounted IMU bracket bodies)
byte-identical:
  - The floating base is still ``base_link`` (the torso/chest) -- unchanged mass,
    inertia, and mesh, even though it no longer carries arms/head/cameras as
    children. This is a deliberate simplification (documented below), not an
    attempt to redistribute the removed mass back onto the torso.
  - ``hip_link`` still hangs *below* the torso through the single-DOF
    ``waist_joint`` -- kept, since it is the torso<->pelvis joint, not an
    upper-body joint.
  - Both 6-DOF legs (hip roll/yaw/pitch, knee pitch, ankle pitch/roll) are
    unchanged.
  - Removed entirely: both 7-DOF arms, the 3-DOF head, and the three chest-camera
    dummy bodies.

Every joint range, effort limit, and PD gain below is copied verbatim from
``skandha_robot_constants.py`` for the joints that survive -- this is the same
physical leg/waist hardware, just fewer actuated joints, not a re-derivation.

## Why keep the torso body at all?

A floating base needs *some* rigid body to be the free-jointed root and to define
where the legs/waist mount -- ``base_link`` is that body. Its mass (9.206 kg) and
inertia are Skandha's own real torso-shell values (not a KodyRobot substitution,
see the parent robot's README), so it is a physically meaningful "lower body +
torso shell" model, not a placeholder. Whether a *real* lower-body-only test rig
should carry additional ballast mass to represent the removed arms/head is a
hardware question for the robotics team, not something guessed here -- flagged
in ``src/assets/robots/skandhalowerbody/README.md``.

## Total mass

Compiled model total mass: **~45.14 kg** (full Skandha is ~56.2 kg; the ~11 kg
difference is the removed arms + head + chest-camera dummies).

## Home keyframe

Reuses the exact same base height and leg/waist joint angles as full Skandha's
home keyframe (see that file for the forward-kinematics derivation) -- leg
kinematics are unaffected by removing the arms/head, and this was verified
directly: with these same values, all 8 foot-contact-sphere geoms land within
<1 mm of world Z = 0 on this trimmed model. The two ``shoulder_roll`` entries
from the full-body keyframe are dropped since there are no arm joints here.

## Actuators

Unchanged from ``skandha_robot_constants.py`` for every joint group that
survives: hip_roll, hip_yaw, hip_pitch, knee_pitch, ankle_pitch, ankle_roll,
waist. The arm/head/wrist actuator groups are simply not instantiated.
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
SKANDHA_LOWER_BODY_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=(r"waist_joint",),
  stiffness=150.0,
  damping=5.0,
  effort_limit=91.0,
  armature=0.01,
)


##
# Keyframe config.
##

# See module docstring: identical leg/waist values to full Skandha's home
# keyframe, verified flat-footed (<1mm) on this trimmed model. No arm joints
# to set (shoulder_roll_r/l dropped -- they don't exist here).
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.8692443230706399),
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
    SKANDHA_LOWER_BODY_ACTUATOR_WAIST,
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
