#!/usr/bin/env python3
"""
Estimate a mechanically reasonable standing/base pose for KodyRobot.

Uses the actual KodyRobot MuJoCo XML and optimizes the 30 primary
1-DoF revolute/hinge joints.

Usage:
    python estimate_kodyrobot_pose.py
    python estimate_kodyrobot_pose.py --xml path/to/master_assembly_v1_mujoco.xml
    python estimate_kodyrobot_pose.py --iterations 200
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


DEFAULT_XML = (
    Path(__file__).resolve().parent
    / "src/assets/robots/kodyrobot/xmls/master_assembly_v1_mujoco.xml"
)

JOINT_NAMES = [
    # Legs: 12
    "hip_roll_l_joint", "hip_yaw_l_joint", "hip_pitch_l_joint",
    "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint",
    "hip_roll_r_joint", "hip_yaw_r_joint", "hip_pitch_r_joint",
    "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint",

    # Waist: 1
    "waist_yaw_joint",

    # Left arm: 7
    "shoulder_pitch_l_joint", "shoulder_roll_l_joint",
    "shoulder_yaw_l_joint", "elbow_pitch_l_joint",
    "elbow_yaw_l_joint", "wrist_pitch_l_joint", "wrist_roll_l_joint",

    # Right arm: 7
    "shoulder_pitch_r_joint", "shoulder_roll_r_joint",
    "shoulder_yaw_r_joint", "elbow_pitch_r_joint",
    "elbow_yaw_r_joint", "wrist_pitch_r_joint", "wrist_roll_r_joint",

    # Head: 3
    "head_yaw_joint", "head_pitch_joint", "head_roll_joint",
]

LEFT_FOOT_SITE = "left_foot"
RIGHT_FOOT_SITE = "right_foot"

# Current generated KodyRobot HOME_KEYFRAME is all zeros.
HOME_POSE = np.zeros(len(JOINT_NAMES), dtype=float)

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
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    qpos_ids = []
    for name in JOINT_NAMES:
        jid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if jid < 0:
            raise RuntimeError(f"Joint not found in XML: {name}")

        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            raise RuntimeError(
                f"Joint '{name}' is not a 1-DoF hinge/revolute joint."
            )

        qpos_ids.append(int(model.jnt_qposadr[jid]))

    left = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, LEFT_FOOT_SITE
    )
    right = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, RIGHT_FOOT_SITE
    )

    if left < 0 or right < 0:
        raise RuntimeError(
            f"XML must contain sites '{LEFT_FOOT_SITE}' and "
            f"'{RIGHT_FOOT_SITE}'."
        )

    return model, data, np.asarray(qpos_ids), left, right


def set_pose(model, data, q, qpos_ids, base_height=1.03):
    data.qpos[:] = 0.0

    # Free floating base, upright orientation.
    data.qpos[0:3] = [0.0, 0.0, base_height]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

    for i, qpos_id in enumerate(qpos_ids):
        data.qpos[qpos_id] = q[i]

    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def metrics(model, data, left, right):
    lp = data.site_xpos[left].copy()
    rp = data.site_xpos[right].copy()

    total_mass = np.sum(model.body_mass)
    com = np.sum(
        model.body_mass[:, None] * data.xipos,
        axis=0,
    ) / total_mass

    center = 0.5 * (lp + rp)

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
    q, model, data, qpos_ids, left, right, lower, upper
):
    set_pose(model, data, q, qpos_ids)
    m = metrics(model, data, left, right)

    com_cost = m["com_xy"] ** 2
    height_cost = m["height"] ** 2
    tilt_cost = m["left_tilt"] ** 2 + m["right_tilt"] ** 2
    angle_cost = np.mean((q / math.pi) ** 2)

    index = {name: i for i, name in enumerate(JOINT_NAMES)}
    symmetry_cost = 0.0

    for a, b in SYMMETRY_OPPOSITE:
        symmetry_cost += (q[index[a]] + q[index[b]]) ** 2

    for a, b in SYMMETRY_SAME:
        symmetry_cost += (q[index[a]] - q[index[b]]) ** 2

    margin = 0.05
    limit_cost = 0.0

    for i in range(len(q)):
        if q[i] - lower[i] < margin:
            limit_cost += (margin - (q[i] - lower[i])) ** 2
        if upper[i] - q[i] < margin:
            limit_cost += (margin - (upper[i] - q[i])) ** 2

    return float(
        20.0 * com_cost
        + 30.0 * height_cost
        + 10.0 * tilt_cost
        + 0.05 * angle_cost
        + 1.0 * symmetry_cost
        + 10.0 * limit_cost
    )


def bounds_from_model(model):
    bounds = []

    for name in JOINT_NAMES:
        jid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name
        )

        if model.jnt_limited[jid]:
            lo, hi = model.jnt_range[jid]
            margin = min(0.03, 0.1 * (hi - lo))
            bounds.append(
                (float(lo + margin), float(hi - margin))
            )
        else:
            bounds.append((-math.pi, math.pi))

    return np.asarray(bounds, dtype=float)


def print_result(q, m, home_cost, optimized_cost):
    print("\n" + "=" * 76)
    print("ESTIMATED KODYROBOT HOME / STANDING POSE")
    print("=" * 76)

    print("\nJoint positions in RADIANS:")
    print("home_pose={")
    for name, value in zip(JOINT_NAMES, q):
        print(f'    "{name}": {value:+.6f},')
    print("}")

    print("\nDegrees (reference only):")
    for name, value in zip(JOINT_NAMES, q):
        print(
            f"  {name:<25} {math.degrees(value):+8.2f} deg"
        )

    print("\nStanding metrics:")
    print(
        f"  COM              = "
        f"({m['com'][0]:+.4f}, {m['com'][1]:+.4f}, "
        f"{m['com'][2]:+.4f}) m"
    )
    print(
        f"  Support center    = "
        f"({m['center'][0]:+.4f}, {m['center'][1]:+.4f}, "
        f"{m['center'][2]:+.4f}) m"
    )
    print(f"  COM XY error      = {m['com_xy']:.5f} m")
    print(f"  Foot height diff  = {m['height']:.5f} m")
    print(
        f"  Left foot tilt    = "
        f"{math.degrees(m['left_tilt']):.3f} deg"
    )
    print(
        f"  Right foot tilt   = "
        f"{math.degrees(m['right_tilt']):.3f} deg"
    )

    print("\nObjective:")
    print(f"  Existing home     = {home_cost:.8f}")
    print(f"  Optimized         = {optimized_cost:.8f}")
    print("=" * 76)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xml",
        type=Path,
        default=DEFAULT_XML,
        help="Path to KodyRobot MuJoCo XML.",
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
        default=1.03,
        help="Floating base Z position in meters.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open MuJoCo viewer with the final pose.",
    )
    args = parser.parse_args()

    if not args.xml.exists():
        raise FileNotFoundError(
            f"XML not found: {args.xml}\n"
            "Use --xml to specify the actual KodyRobot XML."
        )

    model, data, qpos_ids, left, right = load_model(args.xml)
    bounds = bounds_from_model(model)

    x0 = np.clip(HOME_POSE, bounds[:, 0], bounds[:, 1])

    print(f"Using XML: {args.xml}")
    print(f"MuJoCo nq={model.nq}, nv={model.nv}")
    print(
        f"Optimizing {len(JOINT_NAMES)} KodyRobot "
        "joint positions in radians..."
    )

    result = differential_evolution(
        lambda q: objective(
            q, model, data, qpos_ids, left, right,
            bounds[:, 0], bounds[:, 1]
        ),
        bounds=[tuple(x) for x in bounds],
        maxiter=args.iterations,
        popsize=10,
        seed=42,
        polish=False,
        workers=1,
    )

    refined = minimize(
        lambda q: objective(
            q, model, data, qpos_ids, left, right,
            bounds[:, 0], bounds[:, 1]
        ),
        result.x,
        method="L-BFGS-B",
        bounds=[tuple(x) for x in bounds],
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    q = np.clip(refined.x, bounds[:, 0], bounds[:, 1])

    home_cost = objective(
        x0, model, data, qpos_ids, left, right,
        bounds[:, 0], bounds[:, 1]
    )
    optimized_cost = objective(
        q, model, data, qpos_ids, left, right,
        bounds[:, 0], bounds[:, 1]
    )

    if home_cost <= optimized_cost:
        print(
            "\nExisting zero HOME_POSE scores better; "
            "keeping it."
        )
        q = x0
        optimized_cost = home_cost

    set_pose(
        model, data, q, qpos_ids,
        base_height=args.base_height,
    )
    m = metrics(model, data, left, right)

    print_result(q, m, home_cost, optimized_cost)

    if args.visualize:
        import mujoco.viewer

        print("\nClose the MuJoCo viewer to exit.")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                mujoco.mj_forward(model, data)
                viewer.sync()


if __name__ == "__main__":
    main()
