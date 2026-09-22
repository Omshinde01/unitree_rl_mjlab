"""Skandha constants.

Skandha is a 30-DOF full-size humanoid, integrated from the "no-hands" URDF in
``descriptions_package/`` following the same pattern as ``kodyrobot`` (and, through it,
``unitree_h2``/``unitree_g1``):
  - A floating base at ``base_link`` (the torso/chest), with the pelvis-like
    ``hip_link`` hanging *below* it through a single-DOF ``waist_joint``.
  - 7-DOF arms (shoulder pitch/roll/yaw, elbow pitch/yaw, wrist roll/pitch).
  - 6-DOF legs (hip roll/yaw/pitch, knee pitch, ankle pitch/roll).
  - A 3-DOF head (yaw/pitch/roll).

Skandha's mesh geometry is **byte-identical** to KodyRobot's (verified: matching MD5
hashes on every shared STL, e.g. ``base_link.STL``, ``RA_link_1.STL``, ``hip_link.STL``),
and its kinematic tree (body names, joint positions/orientations, joint limits on every
directly-actuated joint) matches KodyRobot's to 6+ significant figures wherever KodyRobot
did not deliberately override a value. This is the same physical hardware family, not a
coincidence, and several decisions below explicitly rely on that fact -- each is called
out with *why*.

## Parallel-linkage (crank/support) joints

Like KodyRobot, Skandha's source URDF represents several DOF as a duplicated pair: a
"_joint_support" joint carrying a generic, unconstrained placeholder range
(``[-3.14, 3.14]`` or similar) that continues the real kinematic chain to the next
structural link, and a same-position "_joint" (no suffix) carrying the *real* motor
limits/effort but terminating in a disconnected "crank"/"crank_shaft" stub body (a 4-bar
linkage motor stub, not simulated as a closed loop). For 7 of these 9 pairs (wrist_roll_r,
knee_pitch_{r,l}, ankle_pitch_{r,l}, head_roll -- verified via forward kinematics: the
support and crank joints' world-frame axes are exactly parallel/antiparallel at qpos=0),
the pair was resolved the same way KodyRobot's was: the crank/motor joint is deleted (its
body becomes a welded, joint-less stub, kept only for its collision/visual geometry), and
the support joint is renamed to the clean (no "_support") name and given the crank
joint's real range + actuatorfrcrange.

For the remaining 2 pairs (ankle_roll_{r,l}, wrist_pitch_l), the crank joint's world-frame
axis was found to be **~orthogonal** (dot product ~0.00-0.02, not ~1) to the support
joint's axis at qpos=0 -- this is a genuine 2-motor differential mechanism (both motors'
shafts sit roughly parallel in the shin/forearm; one output mode shares the motors' own
axis, the other is realized through the linkage geometry, not either motor's raw spin
axis). Transplanting the crank's numeric limit onto the support joint is not geometrically
meaningful for these three, so a well-justified fallback was used instead (see
``CONSERVATIVE_FALLBACK_RANGE`` in the build script / README): ankle_roll gets a
conservative +-20 deg ROM (narrower than KodyRobot/H2's +-30 deg, pending real hardstop
data), and wrist_pitch_l reuses wrist_pitch_r's own real, verified range directly
(unmirrored -- confirmed empirically that Skandha's own URDF gives wrist_roll_l the exact
same numeric bounds as wrist_roll_r, i.e. this wrist assembly's numbers transfer directly
between sides rather than negating).

## Mass properties

Skandha's own URDF mass/inertia data has an unambiguous data-quality bug: two of the
three chest-camera dummy links report ``mass=20.157kg`` (a depth camera is grams, not
20kg -- a classic default-density CAD-export artifact), and with that bug alone
subtracted, the *entire rest of the robot* (both arms + both legs + head) would total
only ~20kg against a 9.2kg torso -- physically implausible for a full-size humanoid
(torso is normally ~40-50% of body mass, not ~90%). KodyRobot -- confirmed to be the same
CAD geometry -- carries a verified-plausible mass for every one of these same-named
bodies (its own README documents these as coming from a corrected "master_v1.urdf" with
real material assignments, not guessed). Given identical geometry, reusing KodyRobot's
per-body mass (and inertia, rescaled proportionally so KodyRobot's principal-axis *shape*
carries over exactly) is a derived, verified substitution for known-bad data -- not a
blind copy of KodyRobot-specific tuning. ``base_link`` is the one exception: its own
Skandha-URDF mass (9.206kg) is physically plausible on its own (a torso frame this size
plausibly houses this much battery/compute) and KodyRobot's finalized value for it (2kg)
is an undocumented deviation from *its own* raw URDF value with no stated rationale in
the KodyRobot README, so it was not carried over. Resulting total mass: ~56.2kg
(KodyRobot is ~48.9kg; the difference is almost entirely this heavier torso choice).

## Actuators

Every joint-group *effort limit* below is identical, group-for-group, to KodyRobot's
(hip_roll 235, hip_pitch/knee_pitch 330, ankle 55, waist 91, shoulder_pitch 91,
shoulder_roll 95, shoulder_yaw 35, elbow_pitch 35, elbow_yaw 24, wrist/head_yaw/pitch 6.3,
head_roll 6.3) -- these come directly from Skandha's *own* URDF motor-side joint limits
(see the crank/support note above), independently of KodyRobot. That every single value
coincides exactly with KodyRobot's is strong, independently-verifiable evidence of shared
actuator hardware (same motors/gearboxes), which is why KodyRobot's already-tuned
stiffness/damping (empirically verified stable in that repo) is reused here rather than
re-guessed from scratch -- same actuators, same reasonable starting gains, conservative
per the "stable simulation > aggressive control" priority.
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

SKANDHA_ROBOT_XML: Path = (
  SRC_PATH / "assets" / "robots" / "skandharobot" / "xmls" / "SkandhaRobot.xml"
)
assert SKANDHA_ROBOT_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, SKANDHA_ROBOT_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(SKANDHA_ROBOT_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
#
# Stiffness/damping reused from kodyrobot's already-verified values: Skandha's own URDF
# gives *identical* per-joint-group effort limits to kodyrobot (independent evidence of
# shared actuator hardware -- see module docstring), so kodyrobot's stiffness/damping is
# a well-justified starting point, not a guess.
##

SKANDHA_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(r"hip_roll_[rl]_joint",),
  stiffness=180.0,
  damping=6.0,
  effort_limit=235.0,
  armature=0.01,
)
SKANDHA_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(r"hip_yaw_[rl]_joint",),
  stiffness=180.0,
  damping=6.0,
  effort_limit=235.0,
  armature=0.01,
)
SKANDHA_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(r"hip_pitch_[rl]_joint",),
  stiffness=220.0,
  damping=8.0,
  effort_limit=330.0,
  armature=0.01,
)
SKANDHA_ACTUATOR_KNEE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(r"knee_pitch_[rl]_joint",),
  stiffness=250.0,
  damping=8.0,
  effort_limit=330.0,
  armature=0.01,
)
SKANDHA_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(r"ankle_pitch_[rl]_joint",),
  stiffness=60.0,
  damping=2.5,
  effort_limit=55.0,
  armature=0.01,
)
SKANDHA_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(r"ankle_roll_[rl]_joint",),
  stiffness=60.0,
  damping=2.5,
  effort_limit=55.0,
  armature=0.01,
)
SKANDHA_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=(r"waist_joint",),
  stiffness=150.0,
  damping=5.0,
  effort_limit=91.0,
  armature=0.01,
)
SKANDHA_ACTUATOR_SHOULDER_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(r"shoulder_pitch_[rl]_joint",),
  stiffness=90.0,
  damping=4.0,
  effort_limit=91.0,
  armature=0.01,
)
SKANDHA_ACTUATOR_SHOULDER_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(r"shoulder_roll_[rl]_joint",),
  stiffness=85.0,
  damping=4.0,
  effort_limit=95.0,
  armature=0.01,
)
SKANDHA_ACTUATOR_SHOULDER_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(r"shoulder_yaw_[rl]_joint",),
  stiffness=65.0,
  damping=3.0,
  effort_limit=35.0,
  armature=0.01,
)
SKANDHA_ACTUATOR_ELBOW_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(r"elbow_pitch_[rl]_joint",),
  stiffness=60.0,
  damping=3.0,
  effort_limit=35.0,
  armature=0.01,
)
SKANDHA_ACTUATOR_ELBOW_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(r"elbow_yaw_[rl]_joint",),
  stiffness=45.0,
  damping=2.5,
  effort_limit=24.0,
  armature=0.01,
)
SKANDHA_ACTUATOR_WRIST = BuiltinPositionActuatorCfg(
  target_names_expr=(r"wrist_roll_[rl]_joint", r"wrist_pitch_[rl]_joint"),
  stiffness=20.0,
  damping=1.0,
  effort_limit=6.3,
  armature=0.01,
)
SKANDHA_ACTUATOR_HEAD_YAW_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(r"head_yaw_joint", r"head_pitch_joint"),
  stiffness=20.0,
  damping=1.0,
  effort_limit=6.3,
  armature=0.01,
)
SKANDHA_ACTUATOR_HEAD_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(r"head_roll_joint",),
  stiffness=15.0,
  damping=0.8,
  effort_limit=6.3,
  armature=0.01,
)


##
# Keyframe config.
##

# Home keyframe: a slight, self-collision-free, flat-footed crouch derived from Skandha's
# OWN geometry via forward kinematics (mujoco.mj_forward), not guessed and not copied
# from KodyRobot's numeric values:
#   1. hip_pitch sign convention verified by FK (positive hip_pitch_r / negative
#      hip_pitch_l both flex the corresponding leg forward, matching Skandha's own
#      asymmetric joint-range bias, e.g. hip_pitch_r=[-0.88,+1.39] rad is forward-biased).
#   2. Crouch magnitude: hip_pitch = 15% of the smaller of the two legs' own forward-ROM
#      bound (~0.206 rad); knee_pitch = 25% of the smaller of the two legs' own flexion
#      bound (~0.182 rad) -- the SAME magnitude on both sides (not the same fraction of
#      each side's possibly-different range), so the stance is geometrically symmetric.
#   3. ankle_pitch solved numerically per leg (fine grid search + local refinement) to
#      minimize the world-Z spread of the foot collision mesh's lowest vertices --
#      i.e. the actual flat-foot condition, verified to better than 1e-4 rad of tilt.
#   4. shoulder_roll sign for slight arm abduction verified by FK (checked which sign
#      increases lateral distance from the torso centerline on each side).
#   5. Base height (0.8692m) computed directly from the lowest foot-contact-sphere world
#      Z at this leg pose, so both feet land exactly on z=0 at reset.
#   6. Self-collision-checked: a 60-sample random-pose sweep across the full joint range
#      found 7 permanently-overlapping body pairs (tight CAD modeling at a joint, not a
#      real collision risk -- 5 of these exactly match KodyRobot's own documented list,
#      confirming the shared-hardware hypothesis; 2 are new, Skandha-specific findings:
#      each shin/foot-roll-link pair). A further check at the *exact* home-keyframe pose
#      (including the +-0.15 rad arm abduction) found 3 more overlaps specific to the
#      left arm's resting position (LA_link_2/base_link, LA_hand_link/LA_link_{4,5}) --
#      these did not appear in the generic random sweep because they only occur near
#      this particular resting posture, not across the whole joint range. All 10 pairs
#      are excluded via <contact><exclude> in the XML.
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.8692443230706399),
  joint_pos={
    "hip_pitch_r_joint": 0.20573850000000002,
    "hip_pitch_l_joint": -0.20573850000000002,
    "knee_pitch_r_joint": -0.18223825,
    "knee_pitch_l_joint": 0.18223825,
    "ankle_pitch_r_joint": -0.08492277143060875,
    "ankle_pitch_l_joint": 0.0849033936169159,
    "shoulder_roll_r_joint": -0.15,
    "shoulder_roll_l_joint": 0.15,
  },
  joint_vel={".*": 0.0},
)


##
# Collision config.
##

# Same regex convention as kodyrobot (same OEM naming: every crank/motor-stub body name
# contains "crank"), verified against Skandha's own XML:
#   - crank/motor stub bodies (RA_link_6_motor_crank_shaft, LA_link_6_crank_shaft,
#     RL_link_4_crank_shaft, LL_link_4_crank_shaft, RL/LL_foot_{pitch,roll}_crank,
#     head_link_3_crank_shaft) are now welded (joint-less, see module docstring) and
#     permanently embedded in their parent's geometry -- excluded from collision.
#   - feet have no "<link>_collision" mesh in use for ground contact; 4 explicit 5mm
#     sphere geoms per foot (at corners derived from Skandha's own foot collision-hull
#     mesh, not copied from KodyRobot) are used instead, matched by ".*_contact\\d+$".
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

SKANDHA_ROBOT_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    SKANDHA_ACTUATOR_HIP_ROLL,
    SKANDHA_ACTUATOR_HIP_YAW,
    SKANDHA_ACTUATOR_HIP_PITCH,
    SKANDHA_ACTUATOR_KNEE_PITCH,
    SKANDHA_ACTUATOR_ANKLE_PITCH,
    SKANDHA_ACTUATOR_ANKLE_ROLL,
    SKANDHA_ACTUATOR_WAIST,
    SKANDHA_ACTUATOR_SHOULDER_PITCH,
    SKANDHA_ACTUATOR_SHOULDER_ROLL,
    SKANDHA_ACTUATOR_SHOULDER_YAW,
    SKANDHA_ACTUATOR_ELBOW_PITCH,
    SKANDHA_ACTUATOR_ELBOW_YAW,
    SKANDHA_ACTUATOR_WRIST,
    SKANDHA_ACTUATOR_HEAD_YAW_PITCH,
    SKANDHA_ACTUATOR_HEAD_ROLL,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_skandha_robot_cfg() -> EntityCfg:
  """Get a fresh Skandha configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when the config is
  shared across multiple places.
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=SKANDHA_ROBOT_ARTICULATION,
  )


SKANDHA_ROBOT_ACTION_SCALE: dict[str, float] = {}
for a in SKANDHA_ROBOT_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    SKANDHA_ROBOT_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_skandha_robot_cfg())

  viewer.launch(robot.spec.compile())
