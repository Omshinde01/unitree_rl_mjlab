# KodyRobot

A 30-DOF full-size humanoid integrated into this repo's mjlab-based velocity
RL pipeline, following the same pattern as `unitree_h2`/`unitree_g1`.

This document describes what was set up, *why* each choice was made, what
was empirically verified (not assumed), and how to train/monitor/play a
policy. Everything here is robot-specific; no mjlab/core framework files
were modified.

## 1. Robot structure

KodyRobot's floating base is `base_link` — the torso/chest — not a pelvis.
The pelvis-like `hip_link` hangs *below* the torso and connects to it through
a single-DOF `waist_joint`. Legs and head attach to `hip_link`/`base_link`
respectively; arms attach directly to `base_link`.

| Group  | Joints (×2, `_r`/`_l` suffix)                                                | DOF |
|--------|-------------------------------------------------------------------------------|-----|
| Arm    | `shoulder_pitch`, `shoulder_roll`, `shoulder_yaw`, `elbow_pitch`, `elbow_yaw`, `wrist_roll`, `wrist_pitch` | 7×2 = 14 |
| Leg    | `hip_roll`, `hip_yaw`, `hip_pitch`, `knee_pitch`, `ankle_pitch`, `ankle_roll` | 6×2 = 12 |
| Waist  | `waist_joint`                                                                | 1   |
| Head   | `head_yaw`, `head_pitch`, `head_roll`                                        | 3   |
| **Total** |                                                                            | **30** |

Total robot mass (from the compiled model): **≈48.9 kg**.

Naming quirk vs. Unitree robots: joints use an `_r`/`_l` **suffix**
(`hip_pitch_r_joint`), not a `right_`/`left_` **prefix**. More importantly,
several left/right joint pairs are mirrored in **axis direction** rather
than in joint-range sign — `hip_pitch`, `knee_pitch`, and `ankle_pitch` each
need *opposite-signed* values on the left vs. right side to produce the same
physical motion (e.g. `knee_pitch_r` only flexes for values in `[0, 2.39]`,
while `knee_pitch_l` only flexes for values in `[-2.39, 0]`). This was
**verified by running forward kinematics** (`mujoco.mj_forward` + inspecting
body/geom world positions), not assumed from the URDF — guessing wrong here
would make the robot stand/walk asymmetrically.

## 2. Files added/changed

All changes are scoped to KodyRobot's own files plus the two small, additive
registration hooks every robot needs (no existing logic in those hub files
was changed, only new lines appended):

| File | What |
|---|---|
| `xmls/KodyRobot.xml` | Edited (see §3) |
| `kody_robot_constants.py` | **New** — actuators, collisions, home keyframe (mjlab `EntityCfg`) |
| `__init__.py` | **New** — package marker |
| `src/assets/robots/__init__.py` | **Appended** — re-exports `get_kody_robot_cfg`/`KODY_ROBOT_ACTION_SCALE`, same pattern as every other robot |
| `src/tasks/velocity/config/kodyrobot/env_cfgs.py` | **New** — `KodyRobot-Rough`/`KodyRobot-Flat` env configs |
| `src/tasks/velocity/config/kodyrobot/rl_cfg.py` | **New** — PPO runner config |
| `src/tasks/velocity/config/kodyrobot/__init__.py` | **New** — task registration (auto-discovered by `src/tasks/__init__.py`, no hub file edit needed here) |

## 3. What changed in `KodyRobot.xml`, and why

The source MJCF (apparently exported from a URDF) needed several fixes
before it could plug into mjlab's entity/actuator/collision pipeline. Each
was verified by actually compiling the model in MuJoCo, not just visually
inspecting the XML:

1. **Named every collision geom.** ~40 `<geom type="mesh" mesh="X_collision">`
   elements had no `name=` attribute. mjlab's `CollisionCfg` selects geoms by
   name via regex, and an unnamed geom can't be selected (and, with several
   sharing the same empty name, can't even be looked up safely). Added
   `name="X_collision"` to match each mesh name.
2. **Removed the hand-authored `<actuator>` block.** The raw XML had 30
   `<motor>` actuators (torque-only, no PD). mjlab instead adds `<position>`
   actuators programmatically from `EntityArticulationInfoCfg` (see §4) —
   keeping both would create duplicate actuator names at build time. The
   original `<motor>` `ctrlrange` values (the manufacturer torque limits)
   were preserved as `effort_limit` in the new actuator configs.
3. **Added a floating-base IMU site + sensors.** The XML only had sensors
   mounted at a *hip-mounted* VectorNav IMU site (kept, untouched, as
   `imu_gyro`/`imu_acc`/`imu_quat`/...). The velocity task's default
   observation/reward wiring (`src/tasks/velocity/velocity_env_cfg.py`)
   expects sensors named `imu_ang_vel`/`imu_lin_vel`/`imu_lin_acc` mounted at
   the robot's floating base, plus a `root_angmom` subtreeangmom sensor —
   exactly like `unitree_h2.xml`. Added a new `imu` site on `base_link` with
   those four sensors, rather than overriding sensor names per-robot in
   `env_cfgs.py`.
4. **Added `left_foot`/`right_foot` sites.** Needed by `foot_height`,
   `foot_clearance`, and `foot_slip` (these read named sites, not geoms).
   None existed; added one inside each `*_foot_roll_link` body, positioned at
   the centroid of that foot's 4 ground-contact spheres.
5. **Added `<contact><exclude>` for 6 body pairs** whose collision meshes
   permanently overlap regardless of joint configuration (tight CAD modeling
   right at a joint, not a real self-collision risk). Found by compiling the
   model, running `mj_forward` at the zero pose, and inspecting `data.ncon`
   — confirmed to persist across a 400-sample random-pose sweep over the
   full joint range:
   `base_link`↔`head_link_3`, `head_link_1`↔`head_link_3`,
   `RA_link_5`↔`RA_link_7`, `LA_link_5`↔`LA_link_7`,
   `hip_link`↔`RL_link_2`, `hip_link`↔`LL_link_2`.
   Everything else that showed up in that random-pose sweep (e.g. legs
   crossing at extreme hip angles, an elbow folding into the torso) was
   **left alone on purpose** — those are genuine self-collisions the
   physics engine is supposed to resolve and the `self_collisions` reward
   is supposed to penalize, not modeling artifacts.
6. **Zeroed the free joint's `damping`/`armature`/`frictionloss`.** The
   file-level `<default><joint .../></default>` (`damping="0.05"
   armature="0.01" frictionloss="0.2"`) was otherwise silently inherited by
   `floating_base_joint`, adding unintended drag to the 6-DOF root.
7. **Removed the embedded standalone-viewer scaffolding** (a second
   `<asset>`/`<visual>`/`<worldbody>` block with a skybox, a `groundplane`
   material, a light, and a `floor` plane, appended after `</sensor>`).
   mjlab attaches an entity's **entire** spec — including any stray
   worldbody children — into the training scene via `MjSpec.attach()`, and
   the scene/terrain system already provides its own ground, lighting, and
   camera. Left in place, this would have silently attached a second,
   redundant ground plane + light as children of the robot's frame. Removed
   to match the `unitree_h2.xml`/`unitree_g1.xml` convention of a "pure"
   entity file.

None of these required touching mjlab itself — all fixes are either in
KodyRobot's own XML or in its own Python config module.

## 4. Actuators (`kody_robot_constants.py`)

No published motor/gearbox spec sheet exists for KodyRobot (unlike G1's
documented 5020/7520/4010 actuators), so PD stiffness/damping are manually
tuned rather than derived from rotor inertia + gear ratio. Effort limits are
**not** guessed — they come directly from the original MJCF's `<motor>`
`ctrlrange` values (the manufacturer torque limits), which were preserved
even though the `<motor>` elements themselves were removed (see §3.2).

| Joint group | stiffness | damping | effort limit (N·m) | armature |
|---|---|---|---|---|
| hip_roll | 180 | 6.0 | 235 | 0.01 |
| hip_yaw | 180 | 6.0 | 235 | 0.01 |
| hip_pitch | 220 | 8.0 | 330 | 0.01 |
| knee_pitch | 250 | 8.0 | 330 | 0.01 |
| ankle_pitch | 60 | 2.5 | 55 | 0.01 |
| ankle_roll | 60 | 2.5 | 55 | 0.01 |
| waist | 150 | 5.0 | 91 | 0.01 |
| shoulder_pitch | 90 | 4.0 | 91 | 0.01 |
| shoulder_roll | 85 | 4.0 | 95 | 0.01 |
| shoulder_yaw | 65 | 3.0 | 35 | 0.01 |
| elbow_pitch | 60 | 3.0 | 35 | 0.01 |
| elbow_yaw | 45 | 2.5 | 24 | 0.01 |
| wrist (roll+pitch) | 20 | 1.0 | 6.3 | 0.01 |
| head_yaw/pitch | 20 | 1.0 | 6.3 | 0.01 |
| head_roll | 15 | 0.8 | 6.3 | 0.01 |

`KODY_ROBOT_ACTION_SCALE` is derived per-joint as `0.25 * effort_limit /
stiffness` (same formula as H2/G1) and plugged into the `JointPositionActionCfg`
in `env_cfgs.py`, so policy actions are scaled appropriately per joint group
rather than using one global scalar.

## 5. Home keyframe

```python
pos = (0, 0, 0.875)
hip_pitch_r=+0.15   hip_pitch_l=-0.15
knee_pitch_r=+0.30  knee_pitch_l=-0.30
ankle_pitch_r=+0.15 ankle_pitch_l=-0.15
shoulder_roll_r=-0.15  shoulder_roll_l=+0.15
# everything else (hip roll/yaw, ankle roll, waist, shoulder pitch/yaw,
# elbow, wrist, head) = 0
```

This was arrived at through an actual tuning loop, not a single guess:

1. Computed the correct left/right signs via forward kinematics (see §1).
2. A first draft used a deep crouch (`hip_pitch=0.35, knee_pitch=0.70`).
   Verified it was self-collision-free and perfectly flat-footed (both feet'
   4 contact spheres land within ~1e-10 of the same height), but an
   open-loop stability test — holding this pose with pure PD control and
   **zero policy action** — showed the robot tipping over in ~1.2s.
3. **Before concluding this was a bug**, the identical test was run against
   `unitree_g1` (a robot config known to train successfully in this repo).
   G1 falls over in a similar ~1.5s under the same zero-action test, despite
   using *much lower*, physically-derived PD stiffness (e.g. knee stiffness
   ≈99 vs. KodyRobot's 250). This is expected: a humanoid held by
   independent per-joint PD with **no active balance feedback** is only
   ever metastable for a second or so — actively balancing is exactly what
   the RL policy is trained to learn, not something the keyframe or PD gains
   are meant to provide on their own.
4. The deep crouch was nonetheless needlessly demanding (more hold-torque
   than necessary, for no stability benefit), so it was replaced with a
   shallower crouch matching G1/H2 proportions (`hip_pitch=0.15,
   knee_pitch=0.30, ankle_pitch=0.15`). Re-verified self-collision-free and
   flat-footed at a recomputed base height (`0.875` instead of `0.837`).
   Under the same zero-action test it now stays within ~5cm of the keyframe
   height for the first ~0.3s (vs. sagging immediately before) and its
   fall timing is now in the same range as the G1 baseline.
5. Final end-to-end sanity check: both `KodyRobot-Rough` and `KodyRobot-Flat`
   were built as real `ManagerBasedRlEnv`s (8 parallel envs each) and run
   for 100 steps (2s) with small random actions — zero terminations across
   all 16 env instances, all 16 reward terms active, actor/critic
   observation shapes as expected (101/116-dim flat, 288/303-dim rough).

## 6. Collisions

Two regex exceptions to the usual `.*_collision` convention, both discovered
(not assumed) by compiling the model:

- Several bodies carry a small welded (joint-less) `*_crank_shaft` /
  `*_motor_crank_shaft` stub — a disconnected 4-bar-linkage motor stub left
  over from the source URDF (see the XML comment next to each joint this
  replaced). Their collision meshes sit permanently embedded in the
  neighboring link. They're matched by `(?!.*crank)` negative lookahead and
  therefore excluded from collision entirely (mjlab's `CollisionCfg`
  disables every geom that *doesn't* match the configured pattern).
- The feet have **no** `*_collision` mesh; ground contact is via 4 explicit
  5mm sphere geoms per foot (`*_contactN`), matched by a separate
  `.*_contact\d+$` pattern and given `condim=3, priority=1, friction=0.6`
  (vs. `condim=1` for general self-collision meshes) — identical philosophy
  to H2/G1's foot-capsule treatment.

`FULL_COLLISION` (self-collision on) is used by default; `FULL_COLLISION_WITHOUT_SELF`
and `FEET_ONLY_COLLISION` are also defined (unused by default) for anyone who
wants to experiment with disabling self-collision for speed.

## 7. Registered tasks

```
KodyRobot-Rough   # generated rough terrain, terrain curriculum, height scan
KodyRobot-Flat    # flat plane, no height scan
```

Both follow the exact same reward/observation/event wiring as
`Unitree-H2-*`/`Unitree-G1-*` (see `src/tasks/velocity/velocity_env_cfg.py`
for the shared defaults), with KodyRobot-specific overrides for:
root body (`base_link`), foot sites/geoms, action scale, and `pose` reward
per-joint-category standard deviations (extended to cover `elbow_yaw` and
`head_*`, which H2/G1 don't have).

## 8. Training

```bash
# Flat terrain (start here — faster iteration, good for sanity-checking the setup)
python scripts/train.py KodyRobot-Flat --env.scene.num-envs=4096

# Rough terrain (after flat-ground walking is solid)
python scripts/train.py KodyRobot-Rough --env.scene.num-envs=4096

# Multi-GPU
python scripts/train.py KodyRobot-Flat --gpu-ids 0 1 --env.scene.num-envs=4096
```

Logs/checkpoints land in `logs/rsl_rl/kodyrobot_velocity/<date_time>/`.

### Monitoring a run

- `cfg.metrics["mean_action_acc"]` and the 16 reward terms (see §7) are
  logged per-iteration by the RSL-RL runner — watch `pose`,
  `track_linear_velocity`, and `self_collisions` first: a healthy run should
  show `pose` and velocity-tracking rewards climbing while
  `self_collisions`/`foot_slip`/`joint_pos_limits` stay near zero.
- If `is_terminated` (`fell_over`, 70° tilt) dominates episode returns for a
  long time after the curriculum ramps `command_vel` up (at iteration
  `5000 * num_steps_per_env`, i.e. step `5000*24`), consider lowering the
  velocity command range ramp or re-checking the `pose` reward's
  `std_walking`/`std_running` tolerances in `env_cfgs.py`.
- `scripts/visualize_terrain.py` and MuJoCo's own viewer
  (`python -m src.assets.robots.kodyrobot.kody_robot_constants`, which
  launches `mujoco.viewer` on the compiled entity) are useful for visually
  sanity-checking the keyframe/collisions before a long training run.

### Playing back a trained policy

```bash
python scripts/play.py KodyRobot-Flat --checkpoint-file=<path-to-model.pt>
```

## 9. Known, expected limitation

Holding the home keyframe with **pure PD control and zero policy action**
(no balance controller) is only metastable for roughly 1–1.5 seconds before
the robot tips over — this was measured and is **true of the reference
`unitree_g1` config as well** (see §5, step 3), so it is not specific to
KodyRobot and is not something further PD tuning is expected to fully
eliminate. It reflects a basic property of this framework's approach: the
*policy* is what learns to balance (weight-shifting, stepping, ankle/hip
strategy), not the default pose or its PD gains. `episode_length_s=20.0`,
frequent `reset_base`/`reset_robot_joints` events, and the dense
`pose`/`body_orientation_l2`/velocity-tracking rewards are what give the
policy enough signal to learn this quickly from a near-random start.
