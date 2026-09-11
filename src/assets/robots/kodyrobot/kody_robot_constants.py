"""KodyRobot constants.

KodyRobot is a full-size humanoid with:
  - A floating base at ``base_link`` (the torso/chest), with the pelvis-like
    ``hip_link`` hanging below it through a single-DOF ``waist_joint``.
  - 7-DOF arms (shoulder pitch/roll/yaw, elbow pitch/yaw, wrist roll/pitch).
  - 6-DOF legs (hip roll/yaw/pitch, knee pitch, ankle pitch/roll).
  - A 3-DOF head (yaw/pitch/roll).

Joint naming uses an ``_r``/``_l`` suffix (not a ``right_``/``left_`` prefix
like Unitree robots), and critically, the left/right kinematic chains are
*mirrored in axis direction* rather than in joint-range sign: for several
joint pairs (hip_pitch, knee_pitch, ankle_pitch) a physically symmetric stance
requires *opposite-signed* values on the left vs. right side. This was
verified empirically via forward kinematics (see the home-keyframe note
below) rather than assumed from URDF axis conventions, since guessing wrong
here would make the robot stand asymmetrically or fall immediately.

The raw MJCF (``KodyRobot.xml``) only defines bodies/joints/geoms/sensors; no
hand-authored ``<actuator>`` block exists there. Actuators are added
programmatically below via ``BuiltinPositionActuatorCfg`` (one <position>
actuator per joint group), exactly like ``unitree_h2``/``unitree_g1``.
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
  SRC_PATH / "assets" / "robots" / "kodyrobot" / "xmls" / "KodyRobot.xml"
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
# Stiffness/damping are manually tuned (no published motor/gearbox specs are
# available for KodyRobot, unlike G1's documented 5020/7520/4010 actuators).
# Effort limits come directly from the MJCF's original <motor> ctrlrange
# values (the manufacturer-provided torque limits), which are preserved here
# even though the raw <motor> actuators themselves were removed from the XML.
##

KODY_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(r"hip_roll_[rl]_joint",),
  stiffness=180.0,
  damping=6.0,
  effort_limit=235.0,
  armature=0.01,
)
KODY_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(r"hip_yaw_[rl]_joint",),
  stiffness=180.0,
  damping=6.0,
  effort_limit=235.0,
  armature=0.01,
)
KODY_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(r"hip_pitch_[rl]_joint",),
  stiffness=220.0,
  damping=8.0,
  effort_limit=330.0,
  armature=0.01,
)
KODY_ACTUATOR_KNEE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(r"knee_pitch_[rl]_joint",),
  stiffness=250.0,
  damping=8.0,
  effort_limit=330.0,
  armature=0.01,
)
KODY_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(r"ankle_pitch_[rl]_joint",),
  stiffness=60.0,
  damping=2.5,
  effort_limit=55.0,
  armature=0.01,
)
KODY_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(r"ankle_roll_[rl]_joint",),
  stiffness=60.0,
  damping=2.5,
  effort_limit=55.0,
  armature=0.01,
)
KODY_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=(r"waist_joint",),
  stiffness=150.0,
  damping=5.0,
  effort_limit=91.0,
  armature=0.01,
)
KODY_ACTUATOR_SHOULDER_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(r"shoulder_pitch_[rl]_joint",),
  stiffness=90.0,
  damping=4.0,
  effort_limit=91.0,
  armature=0.01,
)
KODY_ACTUATOR_SHOULDER_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(r"shoulder_roll_[rl]_joint",),
  stiffness=85.0,
  damping=4.0,
  effort_limit=95.0,
  armature=0.01,
)
KODY_ACTUATOR_SHOULDER_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(r"shoulder_yaw_[rl]_joint",),
  stiffness=65.0,
  damping=3.0,
  effort_limit=35.0,
  armature=0.01,
)
KODY_ACTUATOR_ELBOW_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(r"elbow_pitch_[rl]_joint",),
  stiffness=60.0,
  damping=3.0,
  effort_limit=35.0,
  armature=0.01,
)
KODY_ACTUATOR_ELBOW_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(r"elbow_yaw_[rl]_joint",),
  stiffness=45.0,
  damping=2.5,
  effort_limit=24.0,
  armature=0.01,
)
KODY_ACTUATOR_WRIST = BuiltinPositionActuatorCfg(
  target_names_expr=(r"wrist_roll_[rl]_joint", r"wrist_pitch_[rl]_joint"),
  stiffness=20.0,
  damping=1.0,
  effort_limit=6.3,
  armature=0.01,
)
KODY_ACTUATOR_HEAD_YAW_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(r"head_yaw_joint", r"head_pitch_joint"),
  stiffness=20.0,
  damping=1.0,
  effort_limit=6.3,
  armature=0.01,
)
KODY_ACTUATOR_HEAD_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(r"head_roll_joint",),
  stiffness=15.0,
  damping=0.8,
  effort_limit=6.3,
  armature=0.01,
)


##
# Keyframe config.
##

# Home keyframe: a slightly-crouched, self-collision-free standing pose.
#
# Verified by direct forward-kinematics simulation (not guessed from URDF
# axis signs) using `mujoco.MjModel`/`mj_forward`:
#   - hip_pitch_r=+0.15 / hip_pitch_l=-0.15 flex both legs forward by the same
#     physical amount (the two sides are mirrored in *axis direction*, not in
#     joint-value sign, so a naive symmetric value would bend the legs in
#     opposite physical directions).
#   - knee_pitch_r range is [0, 2.39] (flexion is positive-only) while
#     knee_pitch_l range is [-2.39, 0] (flexion is negative-only); +0.30/-0.30
#     produce matching physical knee bend on both sides.
#   - ankle_pitch_r=+0.15 / ankle_pitch_l=-0.15 keep both feet flat and at the
#     exact same height (verified: right/left foot contact-sphere z-spread is
#     ~1e-10, i.e. perfectly flat).
#   - With this leg pose, `pos.z=0.875` places all 8 foot contact spheres
#     (4 per foot) exactly on the z=0 ground plane.
#   - shoulder_roll_r=-0.15 / shoulder_roll_l=+0.15 abduct the arms slightly
#     away from the torso (confirmed via FK: this is the sign that increases
#     shoulder-to-centerline distance on each side, not decreases it).
# All other joints (hip roll/yaw, ankle roll, waist, shoulder pitch/yaw,
# elbow, wrist, head) are left at 0, which was confirmed self-collision-free.
#
# Crouch depth note: an earlier draft used a much deeper crouch (hip_pitch
# 0.35 / knee_pitch 0.70). A direct open-loop settle test (hold this pose with
# pure PD control, no policy action) showed it falling over in ~1.2s -- but
# the SAME test on the reference unitree_g1 robot (a config known to train
# successfully in this repo) falls over in a similar ~1.5s despite G1 using a
# much shallower crouch (hip_pitch -0.1 / knee_pitch 0.3) and much *lower*
# physically-derived PD stiffness. This confirms pure PD holding is expected
# to be only metastable for a humanoid with no active balance feedback -- the
# RL policy is responsible for learning to balance, not the keyframe -- but
# it also showed the deep crouch was needlessly demanding more hold-torque
# than necessary with no stability benefit. This shallower crouch (matching
# G1/H2 proportions) keeps the same self-collision-free, flat-footed
# guarantees while requiring less torque to hold near the keyframe.
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.875),
  joint_pos={
    "hip_pitch_r_joint": 0.15,
    "hip_pitch_l_joint": -0.15,
    "knee_pitch_r_joint": 0.30,
    "knee_pitch_l_joint": -0.30,
    "ankle_pitch_r_joint": 0.15,
    "ankle_pitch_l_joint": -0.15,
    "shoulder_roll_r_joint": -0.15,
    "shoulder_roll_l_joint": 0.15,
  },
  joint_vel={".*": 0.0},
)


##
# Collision config.
##

# KodyRobot's collision meshes follow a "<link>_collision" naming convention
# (unlike Unitree's "<side>_foot{1-7}_collision" capsule convention), with two
# important exceptions discovered by probing the compiled model:
#
#   1. Several bodies carry a small welded (joint-less) "crank_shaft" /
#      "motor_crank_shaft" stub body representing a disconnected 4-bar-linkage
#      motor stub from the source URDF (see the XML comments next to each).
#      Their collision meshes sit permanently embedded in the neighboring
#      link's geometry -- they are excluded from collision entirely (matched
#      geoms are enabled, everything else is disabled by `disable_other_geoms`).
#   2. The feet have no "<link>_collision" mesh at all; ground contact is
#      via 4 explicit sphere geoms per foot named "..._contactN".
#
# Both were verified with a static + randomized-pose self-collision probe
# (see PR/commit description) before being encoded here. Six additional body
# pairs were found to permanently overlap at every joint configuration
# (tight CAD modeling around a joint, not a real self-collision risk); those
# are excluded via <contact><exclude> in the XML itself, not here.
_COLLISION_GEOM_EXPR = (
  r"^(?!.*crank).*_collision$",
  r".*_contact\d+$",
)

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot contacts are given condim=3.
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

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
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

KODY_ROBOT_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    KODY_ACTUATOR_HIP_ROLL,
    KODY_ACTUATOR_HIP_YAW,
    KODY_ACTUATOR_HIP_PITCH,
    KODY_ACTUATOR_KNEE_PITCH,
    KODY_ACTUATOR_ANKLE_PITCH,
    KODY_ACTUATOR_ANKLE_ROLL,
    KODY_ACTUATOR_WAIST,
    KODY_ACTUATOR_SHOULDER_PITCH,
    KODY_ACTUATOR_SHOULDER_ROLL,
    KODY_ACTUATOR_SHOULDER_YAW,
    KODY_ACTUATOR_ELBOW_PITCH,
    KODY_ACTUATOR_ELBOW_YAW,
    KODY_ACTUATOR_WRIST,
    KODY_ACTUATOR_HEAD_YAW_PITCH,
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
  for n in names:
    KODY_ROBOT_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_kody_robot_cfg())

  viewer.launch(robot.spec.compile())
