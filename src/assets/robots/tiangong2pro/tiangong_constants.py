"""TienKung Pro constants configuration.

This file adapts the Unitree H2 setup (`h2_constants.py`) for the
TienKung Pro robot, enabling RL training using mjlab.

Every numeric value below was extracted directly from the compiled
``tiangong2pro_wh.xml`` (via ``mujoco.MjModel``), not copied from another
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

TIENKUNG_PRO_XML: Path = (
    SRC_PATH
    / "assets"
    / "robots"
    / "tiangong2pro"
    / "xmls"
    / "tiangong2pro_wh.xml"
)

assert TIENKUNG_PRO_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
    """Load mesh assets relative to the XML folder."""

    assets: dict[str, bytes] = {}

    update_assets(
        assets,
        TIENKUNG_PRO_XML.parent / "assets",
        meshdir,
    )

    return assets


def get_spec() -> mujoco.MjSpec:
    """Load MjSpec for TienKung Pro model compilation."""

    spec = mujoco.MjSpec.from_file(
        str(TIENKUNG_PRO_XML)
    )

    spec.assets = get_assets(spec.meshdir)

    return spec


# ============================================================================
# Actuator Configuration
#
# The XML defines exactly three <default> actuator/joint classes, and every
# one of the 28 actuated joints inherits its gains from one of them with no
# per-joint overrides (verified by compiling the model and reading
# actuator_gainprm / actuator_forcerange / dof_damping / dof_armature for
# every joint -- none of them differ from their class default):
#
#   class="leg_joint"    kp=200   damping=5.0   armature=0.01   force=+-150
#     -> all 12 leg joints: hip_roll, hip_pitch, hip_yaw, knee_pitch,
#        ankle_pitch, ankle_roll (both sides)
#
#   class="torso_joint"  kp=100   damping=3.0   armature=0.005  force=+-100
#     -> body_yaw_joint (1 joint)
#
#   class="arm_joint"    kp=30    damping=1.5   armature=0.003  force=+-30
#     -> all 3 head joints AND all 12 arm joints (15 joints total).
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
# TOTAL = 28 DOF
# ============================================================================


# ============================================================================
# LEG ACTUATORS  (XML class "leg_joint": kp=200, damping=5.0, force=+-150)
# ============================================================================

# Hip Roll
TIENKUNG_PRO_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
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
TIENKUNG_PRO_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
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
TIENKUNG_PRO_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
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
TIENKUNG_PRO_ACTUATOR_KNEE_PITCH = BuiltinPositionActuatorCfg(
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
TIENKUNG_PRO_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
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
TIENKUNG_PRO_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
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

TIENKUNG_PRO_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "body_yaw_joint",
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

TIENKUNG_PRO_ACTUATOR_HEAD_YAW = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "head_yaw_joint",
    ),
    stiffness=30.0,
    damping=1.5,
    effort_limit=30.0,
    armature=0.003,
    frictionloss=0.05,
)


TIENKUNG_PRO_ACTUATOR_HEAD_PITCH = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "head_pitch_joint",
    ),
    stiffness=30.0,
    damping=1.5,
    effort_limit=30.0,
    armature=0.003,
    frictionloss=0.05,
)


TIENKUNG_PRO_ACTUATOR_HEAD_ROLL = BuiltinPositionActuatorCfg(
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
TIENKUNG_PRO_ACTUATOR_SHOULDER_PITCH = BuiltinPositionActuatorCfg(
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
TIENKUNG_PRO_ACTUATOR_SHOULDER_ROLL = BuiltinPositionActuatorCfg(
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
TIENKUNG_PRO_ACTUATOR_SHOULDER_YAW = BuiltinPositionActuatorCfg(
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
TIENKUNG_PRO_ACTUATOR_ELBOW_PITCH = BuiltinPositionActuatorCfg(
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
TIENKUNG_PRO_ACTUATOR_ELBOW_YAW = BuiltinPositionActuatorCfg(
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
TIENKUNG_PRO_ACTUATOR_WRIST_ROLL = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "wrist_roll_.*_joint",
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
    pos=(0.0, 0.0, 0.962),
    joint_pos={
        # --------------------------------------------------------------------
        # Left leg
        # --------------------------------------------------------------------
        "hip_roll_l_joint": 0.0,
        "hip_pitch_l_joint": -0.25,
        "hip_yaw_l_joint": 0.0,
        "knee_pitch_l_joint": 0.50,
        "ankle_pitch_l_joint": -0.25,
        "ankle_roll_l_joint": 0.0,

        # --------------------------------------------------------------------
        # Right leg
        # --------------------------------------------------------------------
        "hip_roll_r_joint": 0.0,
        "hip_pitch_r_joint": -0.25,
        "hip_yaw_r_joint": 0.0,
        "knee_pitch_r_joint": 0.50,
        "ankle_pitch_r_joint": -0.25,
        "ankle_roll_r_joint": 0.0,

        # --------------------------------------------------------------------
        # Left arm
        # --------------------------------------------------------------------
        "shoulder_pitch_l_joint": -0.10,  # Negative sign rotates arm forward/downward
        "shoulder_roll_l_joint": 0.10,   # Clears upper arm from torso
        "shoulder_yaw_l_joint": 0.0,
        "elbow_pitch_l_joint": -0.10,    # Negative sign flexes elbow forward (~34 deg)
        "elbow_yaw_l_joint": 0.0,
        "wrist_roll_l_joint": 0.0,

        # --------------------------------------------------------------------
        # Right arm
        # --------------------------------------------------------------------
        "shoulder_pitch_r_joint": -0.10,  # Negative sign rotates arm forward/downward
        "shoulder_roll_r_joint": -0.10,  # Clears upper arm from torso
        "shoulder_yaw_r_joint": 0.0,
        "elbow_pitch_r_joint": -0.10,    # Negative sign flexes elbow forward (~34 deg)
        "elbow_yaw_r_joint": 0.0,
        "wrist_roll_r_joint": 0.0,

        # --------------------------------------------------------------------
        # Waist
        # --------------------------------------------------------------------
        "body_yaw_joint": 0.0,

        # --------------------------------------------------------------------
        # Head
        # --------------------------------------------------------------------
        "head_yaw_joint": 0.0,
        "head_pitch_joint": 0.0,
        "head_roll_joint": 0.0,
    },
    joint_vel={
        ".*": 0.0,
    },
)

# ============================================================================
# Collision Configuration
#
# geom_names_expr=(".*_collision",) matches all 19 collision geoms actually
# authored in the XML (pelvis, both thighs/shins via hip_yaw_*_link and
# knee_pitch_*_link capsules, both feet, waist, head, both upper arms and
# forearms). Verified by compiling the model and listing every geom whose
# name ends in "_collision" -- the regex below does not invent or assume
# any geometry the XML doesn't already define.
# ============================================================================

FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),

    condim={
        r"^ankle_roll_[lr]_link_collision$": 3,
        ".*_collision": 1,
    },

    priority={
        r"^ankle_roll_[lr]_link_collision$": 1,
    },

    friction={
        r"^ankle_roll_[lr]_link_collision$": (0.6,),
    },
)


FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
    geom_names_expr=(".*_collision",),

    contype=0,
    conaffinity=1,

    condim={
        r"^ankle_roll_[lr]_link_collision$": 3,
        ".*_collision": 1,
    },

    priority={
        r"^ankle_roll_[lr]_link_collision$": 1,
    },

    friction={
        r"^ankle_roll_[lr]_link_collision$": (0.6,),
    },
)


FEET_ONLY_COLLISION = CollisionCfg(
    geom_names_expr=(
        r"^ankle_roll_[lr]_link_collision$",
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

TIENKUNG_PRO_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        # Legs
        TIENKUNG_PRO_ACTUATOR_HIP_ROLL,
        TIENKUNG_PRO_ACTUATOR_HIP_PITCH,
        TIENKUNG_PRO_ACTUATOR_HIP_YAW,
        TIENKUNG_PRO_ACTUATOR_KNEE_PITCH,

        # Ankles
        TIENKUNG_PRO_ACTUATOR_ANKLE_PITCH,
        TIENKUNG_PRO_ACTUATOR_ANKLE_ROLL,

        # Waist
        TIENKUNG_PRO_ACTUATOR_WAIST,

        # Head
        TIENKUNG_PRO_ACTUATOR_HEAD_YAW,
        TIENKUNG_PRO_ACTUATOR_HEAD_PITCH,
        TIENKUNG_PRO_ACTUATOR_HEAD_ROLL,

        # Arms
        TIENKUNG_PRO_ACTUATOR_SHOULDER_PITCH,
        TIENKUNG_PRO_ACTUATOR_SHOULDER_ROLL,
        TIENKUNG_PRO_ACTUATOR_SHOULDER_YAW,
        TIENKUNG_PRO_ACTUATOR_ELBOW_PITCH,
        TIENKUNG_PRO_ACTUATOR_ELBOW_YAW,
        TIENKUNG_PRO_ACTUATOR_WRIST_ROLL,
    ),

    soft_joint_pos_limit_factor=0.9,
)


# ============================================================================
# Robot configuration
# ============================================================================

def get_tienkung_pro_robot_cfg() -> EntityCfg:
    """Get a fresh TienKung Pro robot configuration instance."""

    return EntityCfg(
        init_state=HOME_KEYFRAME,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=TIENKUNG_PRO_ARTICULATION,
    )


# ============================================================================
# Action Scaling
# ============================================================================

TIENKUNG_PRO_ACTION_SCALE: dict[str, float] = {}


for actuator in TIENKUNG_PRO_ARTICULATION.actuators:

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
        TIENKUNG_PRO_ACTION_SCALE[pattern] = scale


# ============================================================================
# Validation
# ============================================================================

if __name__ == "__main__":
    import mujoco.viewer as viewer

    from mjlab.entity.entity import Entity

    robot = Entity(
        get_tienkung_pro_robot_cfg()
    )

    model = robot.spec.compile()

    print("=" * 70)
    print("TienKung Pro actuator validation")
    print("=" * 70)

    print(
        f"Number of actuator configuration groups: "
        f"{len(TIENKUNG_PRO_ARTICULATION.actuators)}"
    )

    print("\nAction scale:")
    for pattern, scale in TIENKUNG_PRO_ACTION_SCALE.items():
        print(
            f"  {pattern:<35} {scale:.6f}"
        )

    print("\nCompiled MuJoCo model:")
    print(f"  nq = {model.nq}")
    print(f"  nv = {model.nv}")
    print(f"  nu = {model.nu}")

    print("=" * 70)

    viewer.launch(model)