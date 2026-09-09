#!/usr/bin/env python3
"""
Estimate a mechanically reasonable standing/base pose for KodyRobot.

This script is matched to the supplied:
    master_assembly_v1_mujoco.xml

The XML contains:
    - floating base: base_link
    - 30 primary 1-DoF hinge joints
    - left_foot site
    - right_foot site

The optimizer prefers:
    - COM near the center between the two foot sites
    - equal foot heights
    - approximately flat feet
    - left/right symmetry
    - distance from joint limits
    - moderate joint angles

Usage:
    python estimate_kodyrobot_pose.py

    python estimate_kodyrobot_pose.py \
        --xml src/assets/robots/kodyrobot/xmls/master_assembly_v1_mujoco.xml

    python estimate_kodyrobot_pose.py --iterations 300

    python estimate_kodyrobot_pose.py --visualize
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import mujoco
import numpy as np

try:
    from scipy.optimize import differential_evolution, minimize
except ImportError:
    raise SystemExit("Install scipy first: pip install scipy")


# ---------------------------------------------------------------------------
# XML path
# ---------------------------------------------------------------------------

DEFAULT_XML = (
    Path(__file__).resolve().parent
    / "src/assets/robots/kodyrobot/xmls/master_assembly_v1_mujoco.xml"
)


# ---------------------------------------------------------------------------
# EXACT KodyRobot joint names from the supplied MJCF.
#
# Total = 30 controlled hinge joints:
#   legs  = 12
#   waist = 1
#   arms  = 14
#   head  = 3
# ---------------------------------------------------------------------------

JOINT_NAMES = [
    # Left leg
    "hip_roll_l_joint",
    "hip_yaw_l_joint",
    "hip_pitch_l_joint",
    "knee_pitch_l_joint",
    "ankle_pitch_l_joint",
    "ankle_roll_l_joint",

    # Right leg
    "hip_roll_r_joint",
    "hip_yaw_r_joint",
    "hip_pitch_r_joint",
    "knee_pitch_r_joint",
    "ankle_pitch_r_joint",
    "ankle_roll_r_joint",

    # Waist
    "waist_joint",

    # Left arm
    "shoulder_pitch_l_joint",
    "shoulder_roll_l_joint",
    "shoulder_yaw_l_joint",
    "elbow_pitch_l_joint",
    "elbow_yaw_l_joint",
    "wrist_pitch_l_joint",
    "wrist_roll_l_joint",

    # Right arm
    "shoulder_pitch_r_joint",
    "shoulder_roll_r_joint",
    "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint",
    "elbow_yaw_r_joint",
    "wrist_roll_r_joint",
    "wrist_pitch_r_joint",

    # Head
    "head_yaw_joint",
    "head_pitch_joint",
    "head_roll_joint",
]


LEFT_FOOT_SITE = "left_foot"
RIGHT_FOOT_SITE = "right_foot"


# The supplied XML keyframe is all-zero for the 30 controlled joints.
# The free base is:
#   position = [0, 0, 1]
#   quaternion = [1, 0, 0, 0]
HOME_POSE = np.zeros(len(JOINT_NAMES), dtype=float)


# ---------------------------------------------------------------------------
# Left/right symmetry.
#
# These are based on the joint axes/ranges in the supplied XML.
#
# "same" means q_left ~= q_right
# "opposite" means q_left ~= -q_right
#
# The waist and head are deliberately NOT forced to zero by symmetry.
# They are allowed to move if doing so improves the objective.
# ---------------------------------------------------------------------------

SYMMETRY_OPPOSITE = [
    ("hip_roll_l_joint", "hip_roll_r_joint"),
    ("hip_yaw_l_joint", "hip_yaw_r_joint"),
    ("ankle_roll_l_joint", "ankle_roll_r_joint"),

    ("shoulder_roll_l_joint", "shoulder_roll_r_joint"),
    ("shoulder_yaw_l_joint", "shoulder_yaw_r_joint"),
    ("elbow_yaw_l_joint", "elbow_yaw_r_joint"),
    ("wrist_roll_l_joint", "wrist_roll_r_joint"),
]

SYMMETRY_SAME = [
    ("hip_pitch_l_joint", "hip_pitch_r_joint"),
    ("knee_pitch_l_joint", "knee_pitch_r_joint"),
    ("ankle_pitch_l_joint", "ankle_pitch_r_joint"),

    ("shoulder_pitch_l_joint", "shoulder_pitch_r_joint"),
    ("elbow_pitch_l_joint", "elbow_pitch_r_joint"),
    ("wrist_pitch_l_joint", "wrist_pitch_r_joint"),
]


def load_model(xml_path: Path):
    """Load the exact supplied KodyRobot MJCF."""
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    qpos_ids = []

    for name in JOINT_NAMES:
        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        if jid < 0:
            raise RuntimeError(f"Joint not found in XML: {name}")

        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            raise RuntimeError(
                f"Joint '{name}' is not a hinge joint. "
                "This pose optimizer expects 1-DoF revolute joints."
            )

        qpos_ids.append(int(model.jnt_qposadr[jid]))

    left = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        LEFT_FOOT_SITE,
    )

    right = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        RIGHT_FOOT_SITE,
    )

    if left < 0:
        raise RuntimeError(f"Site not found in XML: {LEFT_FOOT_SITE}")

    if right < 0:
        raise RuntimeError(f"Site not found in XML: {RIGHT_FOOT_SITE}")

    return model, data, np.asarray(qpos_ids), left, right


def set_pose(model, data, q, qpos_ids, base_height=1.0):
    """
    Set KodyRobot floating-base pose.

    The supplied XML places base_link at z=1.0, so 1.0 m is the default.
    The optimizer changes only the 30 controlled hinge joint positions.
    """
    data.qpos[:] = 0.0

    # Free joint qpos:
    # [x, y, z, qw, qx, qy, qz]
    data.qpos[0:3] = [0.0, 0.0, base_height]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

    for i, qpos_id in enumerate(qpos_ids):
        data.qpos[qpos_id] = q[i]

    data.qvel[:] = 0.0

    mujoco.mj_forward(model, data)


def metrics(model, data, left, right):
    """Calculate standing metrics from the actual MuJoCo state."""
    lp = data.site_xpos[left].copy()
    rp = data.site_xpos[right].copy()

    total_mass = np.sum(model.body_mass)

    com = np.sum(
        model.body_mass[:, None] * data.xipos,
        axis=0,
    ) / total_mass

    center = 0.5 * (lp + rp)

    # The foot site's local +Z axis should point upward for a flat foot.
    lz = data.site_xmat[left].reshape(3, 3)[:, 2]
    rz = data.site_xmat[right].reshape(3, 3)[:, 2]

    ltilt = math.acos(np.clip(lz[2], -1.0, 1.0))
    rtilt = math.acos(np.clip(rz[2], -1.0, 1.0))

    return {
        "left": lp,
        "right": rp,
        "com": com,
        "center": center,
        "com_xy": float(np.linalg.norm(com[:2] - center[:2])),
        "height": float(abs(lp[2] - rp[2])),
        "left_tilt": float(ltilt),
        "right_tilt": float(rtilt),
    }


def objective(
    q,
    model,
    data,
    qpos_ids,
    left,
    right,
    lower,
    upper,
):
    """Objective used by the standing-pose optimizer."""
    set_pose(model, data, q, qpos_ids)

    m = metrics(model, data, left, right)

    # 1. Keep COM over the center of the two feet.
    com_cost = m["com_xy"] ** 2

    # 2. Keep both feet at approximately equal height.
    height_cost = m["height"] ** 2

    # 3. Keep both feet approximately flat.
    tilt_cost = (
        m["left_tilt"] ** 2
        + m["right_tilt"] ** 2
    )

    # 4. Prefer moderate joint angles.
    angle_cost = np.mean((q / math.pi) ** 2)

    # 5. Left/right symmetry.
    index = {name: i for i, name in enumerate(JOINT_NAMES)}

    symmetry_cost = 0.0

    for left_name, right_name in SYMMETRY_OPPOSITE:
        a = index[left_name]
        b = index[right_name]
        symmetry_cost += (q[a] + q[b]) ** 2

    for left_name, right_name in SYMMETRY_SAME:
        a = index[left_name]
        b = index[right_name]
        symmetry_cost += (q[a] - q[b]) ** 2

    # 6. Soft penalty for getting too close to a hard joint limit.
    margin = 0.05
    limit_cost = 0.0

    for i in range(len(q)):
        lower_distance = q[i] - lower[i]
        upper_distance = upper[i] - q[i]

        if lower_distance < margin:
            limit_cost += (
                margin - lower_distance
            ) ** 2

        if upper_distance < margin:
            limit_cost += (
                margin - upper_distance
            ) ** 2

    return float(
        20.0 * com_cost
        + 30.0 * height_cost
        + 10.0 * tilt_cost
        + 0.05 * angle_cost
        + 1.0 * symmetry_cost
        + 10.0 * limit_cost
    )


def bounds_from_model(model):
    """Read every joint limit directly from the supplied MJCF."""
    bounds = []

    for name in JOINT_NAMES:
        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        if model.jnt_limited[jid]:
            lo, hi = model.jnt_range[jid]

            # Stay slightly away from the exact mechanical limit.
            margin = min(
                0.03,
                0.1 * (hi - lo),
            )

            bounds.append(
                (
                    float(lo + margin),
                    float(hi - margin),
                )
            )
        else:
            bounds.append(
                (-math.pi, math.pi)
            )

    return np.asarray(bounds, dtype=float)


def print_model_summary(model):
    print("\n" + "-" * 76)
    print("KODYROBOT MODEL")
    print("-" * 76)
    print(f"nq                 : {model.nq}")
    print(f"nv                 : {model.nv}")
    print(f"joints optimized   : {len(JOINT_NAMES)}")
    print(f"total mass         : {np.sum(model.body_mass):.6f} kg")
    print("-" * 76)


def print_joint_limits(model):
    print("\nJoint limits read directly from MuJoCo:")
    print("-" * 76)

    for name in JOINT_NAMES:
        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        lo, hi = model.jnt_range[jid]

        print(
            f"  {name:<25} "
            f"{lo:+.5f} .. {hi:+.5f} rad   "
            f"({math.degrees(lo):+7.2f} .. "
            f"{math.degrees(hi):+7.2f} deg)"
        )


def print_result(q, m, home_cost, optimized_cost):
    print("\n" + "=" * 76)
    print("ESTIMATED KODYROBOT HOME / STANDING POSE")
    print("=" * 76)

    print("\nJoint positions in RADIANS:")
    print("home_pose={")

    for name, value in zip(JOINT_NAMES, q):
        print(
            f'    "{name}": {value:+.6f},'
        )

    print("}")

    print("\nDegrees (reference only):")

    for name, value in zip(JOINT_NAMES, q):
        print(
            f"  {name:<25} "
            f"{math.degrees(value):+8.2f} deg"
        )

    print("\nStanding metrics:")
    print(
        f"  Left foot         = "
        f"({m['left'][0]:+.4f}, "
        f"{m['left'][1]:+.4f}, "
        f"{m['left'][2]:+.4f}) m"
    )

    print(
        f"  Right foot        = "
        f"({m['right'][0]:+.4f}, "
        f"{m['right'][1]:+.4f}, "
        f"{m['right'][2]:+.4f}) m"
    )

    print(
        f"  COM               = "
        f"({m['com'][0]:+.4f}, "
        f"{m['com'][1]:+.4f}, "
        f"{m['com'][2]:+.4f}) m"
    )

    print(
        f"  Support center    = "
        f"({m['center'][0]:+.4f}, "
        f"{m['center'][1]:+.4f}, "
        f"{m['center'][2]:+.4f}) m"
    )

    print(
        f"  COM XY error      = "
        f"{m['com_xy']:.6f} m"
    )

    print(
        f"  Foot height diff  = "
        f"{m['height']:.6f} m"
    )

    print(
        f"  Left foot tilt    = "
        f"{math.degrees(m['left_tilt']):.4f} deg"
    )

    print(
        f"  Right foot tilt   = "
        f"{math.degrees(m['right_tilt']):.4f} deg"
    )

    print("\nObjective:")
    print(f"  Existing HOME     = {home_cost:.10f}")
    print(f"  Optimized         = {optimized_cost:.10f}")

    print("=" * 76)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate a mechanically reasonable "
            "standing/home pose for KodyRobot."
        )
    )

    parser.add_argument(
        "--xml",
        type=Path,
        default=DEFAULT_XML,
        help="Path to master_assembly_v1_mujoco.xml",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=150,
        help="Differential-evolution iterations.",
    )

    parser.add_argument(
        "--base-height",
        type=float,
        default=1.0,
        help=(
            "Floating base Z position in meters. "
            "The supplied XML uses 1.0 m."
        ),
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show the final pose in the MuJoCo viewer.",
    )

    args = parser.parse_args()

    if not args.xml.exists():
        raise FileNotFoundError(
            f"XML not found: {args.xml}\n\n"
            "Use --xml with the actual path, for example:\n"
            "  python estimate_kodyrobot_pose.py "
            "--xml src/assets/robots/kodyrobot/xmls/"
            "master_assembly_v1_mujoco.xml"
        )

    print(f"Using XML: {args.xml}")

    model, data, qpos_ids, left, right = load_model(
        args.xml
    )

    if len(JOINT_NAMES) != 30:
        raise RuntimeError(
            f"Internal error: expected 30 joints, "
            f"but JOINT_NAMES contains {len(JOINT_NAMES)}."
        )

    bounds = bounds_from_model(model)

    # Existing XML HOME keyframe has all 30 controlled joints at zero.
    x0 = np.clip(
        HOME_POSE,
        bounds[:, 0],
        bounds[:, 1],
    )

    print_model_summary(model)
    print_joint_limits(model)

    print(
        f"\nOptimizing {len(JOINT_NAMES)} "
        "KodyRobot joint positions in radians..."
    )

    print("\n[1/2] Differential evolution...")

    result = differential_evolution(
        lambda q: objective(
            q,
            model,
            data,
            qpos_ids,
            left,
            right,
            bounds[:, 0],
            bounds[:, 1],
        ),
        bounds=[
            tuple(x)
            for x in bounds
        ],
        maxiter=args.iterations,
        popsize=10,
        seed=42,
        polish=False,
        workers=1,
        updating="immediate",
    )

    print(
        f"  Best DE objective: "
        f"{result.fun:.10f}"
    )

    print("\n[2/2] Local refinement...")

    refined = minimize(
        lambda q: objective(
            q,
            model,
            data,
            qpos_ids,
            left,
            right,
            bounds[:, 0],
            bounds[:, 1],
        ),
        result.x,
        method="L-BFGS-B",
        bounds=[
            tuple(x)
            for x in bounds
        ],
        options={
            "maxiter": 1000,
            "ftol": 1e-12,
        },
    )

    q = np.clip(
        refined.x,
        bounds[:, 0],
        bounds[:, 1],
    )

    home_cost = objective(
        x0,
        model,
        data,
        qpos_ids,
        left,
        right,
        bounds[:, 0],
        bounds[:, 1],
    )

    optimized_cost = objective(
        q,
        model,
        data,
        qpos_ids,
        left,
        right,
        bounds[:, 0],
        bounds[:, 1],
    )

    # Do not replace a better existing HOME pose with a worse result.
    if home_cost <= optimized_cost:
        print(
            "\nExisting HOME_KEYFRAME scores better "
            "than the optimized pose. Keeping HOME_KEYFRAME."
        )
        q = x0
        optimized_cost = home_cost

    set_pose(
        model,
        data,
        q,
        qpos_ids,
        base_height=args.base_height,
    )

    m = metrics(
        model,
        data,
        left,
        right,
    )

    print_result(
        q,
        m,
        home_cost,
        optimized_cost,
    )

    if args.visualize:
        import mujoco.viewer

        print("\nClose the MuJoCo viewer to exit.")

        with mujoco.viewer.launch_passive(
            model,
            data,
        ) as viewer:
            while viewer.is_running():
                mujoco.mj_forward(model, data)
                viewer.sync()


if __name__ == "__main__":
    main()
