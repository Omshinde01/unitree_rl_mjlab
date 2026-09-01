"""TienKung Pro constants configuration.

This file adapts the Unitree H2 setup (`h2_constants.py`) for the
TienKung Pro robot, enabling RL training using mjlab.

Values are derived from:
1. tiangong2pro_wh.xml
2. TIENKUNG_PRO_CFG
3. h2_constants.py
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
# BuiltinPositionActuatorCfg in the installed mjlab version expects:
#
#     stiffness: float
#     damping: float
#     effort_limit: float
#
# Therefore, joints with different gains/limits are placed into separate
# actuator configuration groups.
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
# LEG ACTUATORS
# ============================================================================

# Hip Roll
TIENKUNG_PRO_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "hip_roll_.*_joint",
    ),
    stiffness=700.0,
    damping=10.0,
    effort_limit=180.0,
    armature=0.01,
)


# Hip Pitch
TIENKUNG_PRO_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "hip_pitch_.*_joint",
    ),
    stiffness=700.0,
    damping=10.0,
    effort_limit=300.0,
    armature=0.01,
)


# Hip Yaw
TIENKUNG_PRO_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "hip_yaw_.*_joint",
    ),
    stiffness=500.0,
    damping=5.0,
    effort_limit=180.0,
    armature=0.01,
)


# Knee Pitch
TIENKUNG_PRO_ACTUATOR_KNEE_PITCH = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "knee_pitch_.*_joint",
    ),
    stiffness=700.0,
    damping=10.0,
    effort_limit=300.0,
    armature=0.01,
)


# ============================================================================
# ANKLE ACTUATORS
# ============================================================================

# Ankle Pitch
TIENKUNG_PRO_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "ankle_pitch_.*_joint",
    ),
    stiffness=200.0,  # was 30.0 — matches XML's leg_joint default kp
    damping=5.0,       # was 2.5 — matches XML's leg_joint joint damping
    effort_limit=150.0,  # was 60.0 — matches XML forcerange (-150, 150)
    armature=0.01,
)


# Ankle Roll
TIENKUNG_PRO_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "ankle_roll_.*_joint",
    ),
    stiffness=200.0,  # was 16.8 — matches XML's leg_joint default kp
    damping=5.0,       # was 1.4 — matches XML's leg_joint joint damping
    effort_limit=150.0,  # was 30.0 — matches XML forcerange (-150, 150)
    armature=0.01,
)


# ============================================================================
# WAIST
# ============================================================================

TIENKUNG_PRO_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "body_yaw_joint",
    ),
    stiffness=20.0,
    damping=1.0,
    effort_limit=50.0,
    armature=0.01,
)


# ============================================================================
# HEAD
# ============================================================================

TIENKUNG_PRO_ACTUATOR_HEAD_YAW = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "head_yaw_joint",
    ),
    stiffness=10.0,
    damping=0.5,
    effort_limit=20.0,
    armature=0.01,
)


TIENKUNG_PRO_ACTUATOR_HEAD_PITCH = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "head_pitch_joint",
    ),
    stiffness=10.0,
    damping=0.5,
    effort_limit=20.0,
    armature=0.01,
)


TIENKUNG_PRO_ACTUATOR_HEAD_ROLL = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "head_roll_joint",
    ),
    stiffness=10.0,
    damping=0.5,
    effort_limit=20.0,
    armature=0.01,
)


# ============================================================================
# ARM ACTUATORS
# ============================================================================

# Shoulder Pitch
TIENKUNG_PRO_ACTUATOR_SHOULDER_PITCH = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "shoulder_pitch_.*_joint",
    ),
    stiffness=60.0,
    damping=3.0,
    effort_limit=52.25,
    armature=0.01,
)


# Shoulder Roll
TIENKUNG_PRO_ACTUATOR_SHOULDER_ROLL = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "shoulder_roll_.*_joint",
    ),
    stiffness=20.0,
    damping=1.5,
    effort_limit=52.25,
    armature=0.01,
)


# Shoulder Yaw
TIENKUNG_PRO_ACTUATOR_SHOULDER_YAW = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "shoulder_yaw_.*_joint",
    ),
    stiffness=10.0,
    damping=1.0,
    effort_limit=35.0,
    armature=0.01,
)


# Elbow Pitch
TIENKUNG_PRO_ACTUATOR_ELBOW_PITCH = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "elbow_pitch_.*_joint",
    ),
    stiffness=10.0,
    damping=1.0,
    effort_limit=35.0,
    armature=0.01,
)


# Elbow Yaw
TIENKUNG_PRO_ACTUATOR_ELBOW_YAW = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "elbow_yaw_.*_joint",
    ),
    stiffness=15.0,
    damping=1.0,
    effort_limit=40.0,
    armature=0.01,
)


# Wrist Roll
TIENKUNG_PRO_ACTUATOR_WRIST_ROLL = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "wrist_roll_.*_joint",
    ),
    stiffness=8.0,
    damping=0.5,
    effort_limit=20.0,
    armature=0.01,
)


# ============================================================================
# Keyframe Configuration
# ============================================================================
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.94),
    joint_pos={
        # --------------------------------------------------------------------
        # Left leg (Unchanged)
        # --------------------------------------------------------------------
        "hip_roll_l_joint": 0.0,
        "hip_pitch_l_joint": -0.25,
        "hip_yaw_l_joint": 0.0,
        "knee_pitch_l_joint": 0.50,
        "ankle_pitch_l_joint": -0.25,
        "ankle_roll_l_joint": 0.0,

        # --------------------------------------------------------------------
        # Right leg (Unchanged)
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
        "shoulder_pitch_l_joint": -0.20,  # Negative sign rotates arm forward/downward
        "shoulder_roll_l_joint": 0.15,   # Clears upper arm from torso
        "shoulder_yaw_l_joint": 0.0,
        "elbow_pitch_l_joint": -0.20,    # Negative sign flexes elbow forward (~34 deg)
        "elbow_yaw_l_joint": 0.0,
        "wrist_roll_l_joint": 0.0,

        # --------------------------------------------------------------------
        # Right arm
        # --------------------------------------------------------------------
        "shoulder_pitch_r_joint": -0.20,  # Negative sign rotates arm forward/downward
        "shoulder_roll_r_joint": -0.15,  # Clears upper arm from torso
        "shoulder_yaw_r_joint": 0.0,
        "elbow_pitch_r_joint": -0.20,    # Negative sign flexes elbow forward (~34 deg)
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
