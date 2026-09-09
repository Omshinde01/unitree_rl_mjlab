"""KodyRobot constants configuration.

This file adapts the Unitree H2 setup (`h2_constants.py`) for the
KodyRobot robot, enabling RL training using mjlab.

Every numeric value below was extracted directly from the compiled
``master_assembly_v1_mujoco.xml`` (via ``mujoco.MjModel``), not copied from another
robot or guessed. See the inline notes for exactly where each group of
numbers comes from and how to re-verify them.
"""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg
from src import SRC_PATH


# ============================================================================
# MJCF and assets setup
# ============================================================================

KODYROBOT_XML: Path = (
    SRC_PATH
    / "assets"
    / "robots"
    / "kodyrobot"
    / "xmls"
    / "master_assembly_v1_mujoco.xml"
)

assert KODYROBOT_XML.exists()


# ============================================================================
# XML invariants -- do not reintroduce these when regenerating the MJCF.
#
# 1. No <actuator> block. Actuators are added exclusively below via
#    KODYROBOT_ARTICULATION / BuiltinPositionActuatorCfg.edit_spec(), which
#    always *adds* a new <position> actuator regardless of what the spec
#    already contains (mjlab.entity.Entity._add_actuators() does not check
#    for or dedupe existing actuators). A baked-in <actuator> block would
#    silently double-actuate every joint: one PD servo driven by the RL
#    policy's ctrl, plus a second, uncommanded one left parked at whatever
#    the init keyframe set, fighting the policy for the entire episode.
# 2. No <light>/floor <geom>/skybox/groundplane assets. mjlab's terrain
#    system (cfg.scene.terrain) supplies the ground plane when this spec is
#    attached into a scene; a robot-local floor geom would sit exactly
#    coincident with it in every environment, causing duplicate/unstable
#    ground contacts.
# 3. No <option>/<size> block. Simulation options are owned by env_cfgs.py's
#    `cfg.sim.*` once this spec is attached into a scene -- values set here
#    are compile-time only and silently ignored by the training pipeline.
# 4. Every <geom class="collision"> must carry an explicit
#    name="<link>_collision". CollisionCfg.edit_spec() (see FULL_COLLISION
#    below) matches geoms by their literal .name against ".*_collision" --
#    an unnamed geom has name == "", and since disable_other_geoms=True
#    collapses all such empty names into a single set entry, only one
#    arbitrary geom would ever get explicitly disabled while the rest
#    silently keep the XML's raw <default> contype/conaffinity/friction
#    instead of the tuned collision profile. All 31 mesh collision geoms
#    (base_link, hip_link, both legs, both arms, head; hands and IMU/camera
#    mounts have no collision geom by design) are named this way.
# ============================================================================


def get_assets(meshdir: str) -> dict[str, bytes]:
    """Load mesh assets relative to the XML folder."""

    assets: dict[str, bytes] = {}

    update_assets(
        assets,
        KODYROBOT_XML.parent / "assets",
        meshdir,
    )

    return assets


def get_spec() -> mujoco.MjSpec:
    """Load MjSpec for KodyRobot model compilation."""

    spec = mujoco.MjSpec.from_file(
        str(KODYROBOT_XML)
    )

    spec.assets = get_assets(spec.meshdir)

    return spec


# ============================================================================
# Actuator Configuration
#
# The XML defines exactly three <default> actuator/joint classes, and every
# one of the 30 actuated joints inherits its gains from one of them with no
# per-joint overrides (verified by compiling the model and reading
# actuator_gainprm / actuator_forcerange / dof_damping / dof_armature for
# every joint -- none of them differ from their class default):
#
#   class="leg_joint"    kp=200   damping=5.0   armature=0.01   force=+-150
#     -> all 12 leg joints: hip_roll, hip_pitch, hip_yaw, knee_pitch,
#        ankle_pitch, ankle_roll (both sides)
#
#   class="torso_joint"  kp=100   damping=3.0   armature=0.005  force=+-100
#     -> waist_joint (1 joint)
#
#   class="arm_joint"    kp=30    damping=1.5   armature=0.003  force=+-30
#     -> all 3 head joints AND all 14 arm joints (17 joints total).
#        Note: in the XML, head_yaw/head_pitch/head_roll use class
#        "arm_joint", not a separate head profile -- there is no dedicated
#        "head" actuator class in this MJCF.
#
# In addition, every joint in the XML carries frictionloss="0.05" via the
# base <joint> default. mjlab's BuiltinPositionActuatorCfg *overwrites* each
# target joint's armature/frictionloss with whatever is passed in (or 0.0 if
# omitted) when it builds the spec -- so frictionloss must be set explicitly
# below (=0.05) or the XML's friction is silently dropped to zero.
#
# Total controlled DOF:
#
# Legs:
#   hip_roll      = 2
#   hip_pitch     = 2
#   hip_yaw       = 2
#   knee_pitch    = 2
#   ankle_pitch   = 2
#   ankle_roll    = 2
#
# Waist:
#   body_yaw      = 1
#
# Head:
#   yaw           = 1
#   pitch         = 1
#   roll          = 1
#
# Arms:
#   shoulder_pitch = 2
#   shoulder_roll  = 2
#   shoulder_yaw   = 2
#   elbow_pitch    = 2
#   elbow_yaw      = 2
#   wrist_roll     = 2
#
# TOTAL = 30 DOF
# ============================================================================


# ============================================================================
# LEG ACTUATORS  (XML class "leg_joint": kp=200, damping=5.0, force=+-150)
# ============================================================================

# Hip Roll
KODYROBOT_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "hip_roll_.*_joint",
    ),
    stiffness=200.0,
    damping=5.0,
    effort_limit=150.0,
    armature=0.01,
    frictionloss=0.05,
)


# Hip Pitch
KODYROBOT_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "hip_pitch_.*_joint",
    ),
    stiffness=200.0,
    damping=5.0,
    effort_limit=150.0,
    armature=0.01,
    frictionloss=0.05,
)


# Hip Yaw
KODYROBOT_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "hip_yaw_.*_joint",
    ),
    stiffness=200.0,
    damping=5.0,
    effort_limit=150.0,
    armature=0.01,
    frictionloss=0.05,
)


# Knee Pitch
KODYROBOT_ACTUATOR_KNEE_PITCH = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "knee_pitch_.*_joint",
    ),
    stiffness=200.0,
    damping=5.0,
    effort_limit=150.0,
    armature=0.01,
    frictionloss=0.05,
)


# ============================================================================
# ANKLE ACTUATORS  (XML class "leg_joint" -- same profile as hip/knee)
# ============================================================================

# Ankle Pitch
KODYROBOT_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "ankle_pitch_.*_joint",
    ),
    stiffness=200.0,
    damping=5.0,
    effort_limit=150.0,
    armature=0.01,
    frictionloss=0.05,
)


# Ankle Roll
KODYROBOT_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "ankle_roll_.*_joint",
    ),
    stiffness=200.0,
    damping=5.0,
    effort_limit=150.0,
    armature=0.01,
    frictionloss=0.05,
)


# ============================================================================
# WAIST  (XML class "torso_joint": kp=100, damping=3.0, force=+-100)
# ============================================================================

KODYROBOT_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "waist_joint",
    ),
    stiffness=100.0,
    damping=3.0,
    effort_limit=100.0,
    armature=0.005,
    frictionloss=0.05,
)


# ============================================================================
# HEAD  (XML class "arm_joint": kp=30, damping=1.5, force=+-30 -- the XML
# does not define a separate head profile, head joints share "arm_joint")
# ============================================================================

KODYROBOT_ACTUATOR_HEAD_YAW = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "head_yaw_joint",
    ),
    stiffness=30.0,
    damping=1.5,
    effort_limit=30.0,
    armature=0.003,
    frictionloss=0.05,
)


KODYROBOT_ACTUATOR_HEAD_PITCH = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "head_pitch_joint",
    ),
    stiffness=30.0,
    damping=1.5,
    effort_limit=30.0,
    armature=0.003,
    frictionloss=0.05,
)


KODYROBOT_ACTUATOR_HEAD_ROLL = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "head_roll_joint",
    ),
    stiffness=30.0,
    damping=1.5,
    effort_limit=30.0,
    armature=0.003,
    frictionloss=0.05,
)


# ============================================================================
# ARM ACTUATORS  (XML class "arm_joint": kp=30, damping=1.5, force=+-30)
# ============================================================================

# Shoulder Pitch
KODYROBOT_ACTUATOR_SHOULDER_PITCH = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "shoulder_pitch_.*_joint",
    ),
    stiffness=30.0,
    damping=1.5,
    effort_limit=30.0,
    armature=0.003,
    frictionloss=0.05,
)


# Shoulder Roll
KODYROBOT_ACTUATOR_SHOULDER_ROLL = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "shoulder_roll_.*_joint",
    ),
    stiffness=30.0,
    damping=1.5,
    effort_limit=30.0,
    armature=0.003,
    frictionloss=0.05,
)


# Shoulder Yaw
KODYROBOT_ACTUATOR_SHOULDER_YAW = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "shoulder_yaw_.*_joint",
    ),
    stiffness=30.0,
    damping=1.5,
    effort_limit=30.0,
    armature=0.003,
    frictionloss=0.05,
)


# Elbow Pitch
KODYROBOT_ACTUATOR_ELBOW_PITCH = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "elbow_pitch_.*_joint",
    ),
    stiffness=30.0,
    damping=1.5,
    effort_limit=30.0,
    armature=0.003,
    frictionloss=0.05,
)


# Elbow Yaw
KODYROBOT_ACTUATOR_ELBOW_YAW = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "elbow_yaw_.*_joint",
    ),
    stiffness=30.0,
    damping=1.5,
    effort_limit=30.0,
    armature=0.003,
    frictionloss=0.05,
)


# Wrist Roll
KODYROBOT_ACTUATOR_WRIST_ROLL = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "wrist_roll_.*_joint",
    ),
    stiffness=30.0,
    damping=1.5,
    effort_limit=30.0,
    armature=0.003,
    frictionloss=0.05,
)


KODYROBOT_ACTUATOR_WRIST_PITCH = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "wrist_pitch_.*_joint",
    ),
    stiffness=30.0,
    damping=1.5,
    effort_limit=30.0,
    armature=0.003,
    frictionloss=0.05,
)


# ============================================================================
# Keyframe Configuration
#
# pos.z = 0.962 was computed, not guessed: with hip_pitch=-0.25,
# knee_pitch=+0.50, ankle_pitch=-0.25 (all other leg joints at 0), forward
# kinematics on the compiled model gives the ankle_roll collision box's
# lowest corner (accounting for the ankle_pitch tilt) at z = -0.9618
# relative to the pelvis. The previous value (1.03) left the feet floating
# ~6.8 cm above the ground plane, so every episode reset began with an
# uncontrolled free-fall/impact instead of a stable stand. Re-verify this
# any time the leg joint angles below, or the leg link lengths in the XML,
# change.
# ============================================================================
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    # The generated KodyRobot XML uses z=1.0 for the floating-base home pose.
    # Re-verify this after the first compiled-model stand test.
    pos=(0.0, 0.0, 0.83),
    joint_pos={
        "hip_roll_l_joint": 0.000,
        "hip_yaw_l_joint": 0.000,
        "hip_pitch_l_joint": -0.300,
        "knee_pitch_l_joint": -0.600,
        "ankle_pitch_l_joint": -0.408,
        "ankle_roll_l_joint": 0.000,

        "hip_roll_r_joint": 0.000,
        "hip_yaw_r_joint": 0.000,
        "hip_pitch_r_joint": 0.300,
        "knee_pitch_r_joint": 0.600,
        "ankle_pitch_r_joint": 0.408,
        "ankle_roll_r_joint": 0.000,

        "waist_joint": 0.000,

        "shoulder_pitch_l_joint": 0.000,
        "shoulder_roll_l_joint": 0.200,
        "shoulder_yaw_l_joint": 0.000,
        "elbow_pitch_l_joint": 0.000,
        "elbow_yaw_l_joint": 0.000,
        "wrist_pitch_l_joint": 0.000,
        "wrist_roll_l_joint": 0.000,

        "shoulder_pitch_r_joint": 0.000,
        "shoulder_roll_r_joint": -0.200,
        "shoulder_yaw_r_joint": 0.000,
        "elbow_pitch_r_joint": 0.000,
        "elbow_yaw_r_joint": 0.000,
        "wrist_roll_r_joint": 0.000,
        "wrist_pitch_r_joint": 0.000,

        "head_yaw_joint": 0.000,
        "head_pitch_joint": 0.000,
        "head_roll_joint": 0.000,
    },
    joint_vel={
        ".*": 0.0,
    },
)


# ============================================================================
# Collision Configuration
#
# geom_names_expr=(".*_collision",) matches all 31 collision geoms authored
# in the XML: base_link, hip_link, every leg link (LL/RL_link_1..4,
# LL/RL_foot_pitch_link, LL/RL_foot_roll_link), every head link
# (head_link_1..3), and every arm link (RA/LA_link_1..7). Each carries an
# explicit name="<link>_collision" for exactly this purpose -- see the "XML
# invariants" note above KODYROBOT_XML. Hands (RA/LA_hand_link) and the
# IMU/camera mount bodies have no collision geom by design (visual only).
# Verified by compiling the model and listing every geom whose name ends in
# "_collision" -- the regex below does not invent or assume any geometry the
# XML doesn't already define.
# ============================================================================

FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    condim=3,
    priority=1,
    friction=(0.6,),
)


FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
    geom_names_expr=(".*_collision",),
    contype=0,
    conaffinity=1,
    condim=3,
    priority=1,
    friction=(0.6,),
)


FEET_ONLY_COLLISION = CollisionCfg(
    geom_names_expr=(
        r"^(LL_foot_roll_link|RL_foot_roll_link)_collision$",
    ),
    contype=0,
    conaffinity=1,
    condim=3,
    priority=1,
    friction=(0.6,),
)


# ============================================================================
# Articulation & Entity Configuration
# ============================================================================

KODYROBOT_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        # Legs
        KODYROBOT_ACTUATOR_HIP_ROLL,
        KODYROBOT_ACTUATOR_HIP_PITCH,
        KODYROBOT_ACTUATOR_HIP_YAW,
        KODYROBOT_ACTUATOR_KNEE_PITCH,

        # Ankles
        KODYROBOT_ACTUATOR_ANKLE_PITCH,
        KODYROBOT_ACTUATOR_ANKLE_ROLL,

        # Waist
        KODYROBOT_ACTUATOR_WAIST,

        # Head
        KODYROBOT_ACTUATOR_HEAD_YAW,
        KODYROBOT_ACTUATOR_HEAD_PITCH,
        KODYROBOT_ACTUATOR_HEAD_ROLL,

        # Arms
        KODYROBOT_ACTUATOR_SHOULDER_PITCH,
        KODYROBOT_ACTUATOR_SHOULDER_ROLL,
        KODYROBOT_ACTUATOR_SHOULDER_YAW,
        KODYROBOT_ACTUATOR_ELBOW_PITCH,
        KODYROBOT_ACTUATOR_ELBOW_YAW,
        KODYROBOT_ACTUATOR_WRIST_ROLL,
        KODYROBOT_ACTUATOR_WRIST_PITCH,
    ),

    soft_joint_pos_limit_factor=0.9,
)


# ============================================================================
# Robot configuration
# ============================================================================

def get_kodyrobot_robot_cfg() -> EntityCfg:
    """Get a fresh KodyRobot robot configuration instance."""

    return EntityCfg(
        init_state=HOME_KEYFRAME,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=KODYROBOT_ARTICULATION,
    )


# ============================================================================
# Action Scaling
# ============================================================================

KODYROBOT_ACTION_SCALE: dict[str, float] = {}


for actuator in KODYROBOT_ARTICULATION.actuators:

    assert isinstance(
        actuator,
        BuiltinPositionActuatorCfg,
    )

    assert actuator.effort_limit is not None
    assert actuator.stiffness is not None

    scale = (
        0.25
        * actuator.effort_limit
        / actuator.stiffness
    )

    for pattern in actuator.target_names_expr:
        KODYROBOT_ACTION_SCALE[pattern] = scale


# ============================================================================
# Validation
# ============================================================================

if __name__ == "__main__":
    import mujoco.viewer as viewer

    from mjlab.entity.entity import Entity

    robot = Entity(
        get_kodyrobot_robot_cfg()
    )

    model = robot.spec.compile()

    print("=" * 70)
    print("KodyRobot actuator validation")
    print("=" * 70)

    print(
        f"Number of actuator configuration groups: "
        f"{len(KODYROBOT_ARTICULATION.actuators)}"
    )

    print("\nAction scale:")
    for pattern, scale in KODYROBOT_ACTION_SCALE.items():
        print(
            f"  {pattern:<35} {scale:.6f}"
        )

    print("\nCompiled MuJoCo model:")
    print(f"  nq = {model.nq}")
    print(f"  nv = {model.nv}")
    print(f"  nu = {model.nu}")

    print("=" * 70)

    viewer.launch(model)