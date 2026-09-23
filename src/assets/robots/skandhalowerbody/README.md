# SkandhaLowerBody

A 13-DOF lower-body-only derivative of `skandharobot`, integrated into this repo's
mjlab-based velocity RL pipeline the same way `skandharobot` itself is. Built by
programmatically stripping the arm, head, and chest-camera subtrees out of the
existing, already-verified `SkandhaRobot.xml` — **the full `skandharobot` integration
was not modified in any way** to produce this robot.

## 0. What "lower body" means here

Kept: the torso (`base_link`, still the floating base), the pelvis-like `hip_link`
hanging below it through `waist_joint`, and both 6-DOF legs.

Removed entirely: both 7-DOF arms, the 3-DOF head, and the three chest-camera dummy
bodies.

| Group | Joints (×2, `_r`/`_l`) | DOF |
|---|---|---|
| Leg | `hip_roll`, `hip_yaw`, `hip_pitch`, `knee_pitch`, `ankle_pitch`, `ankle_roll` | 6×2 = 12 |
| Waist | `waist_joint` | 1 |
| **Total** | | **13** |

Compiled model total mass: **~45.14 kg** (full Skandha is ~56.2 kg; the ~11 kg
difference is the removed arms + head + chest-camera dummies).

`base_link`'s own mass/inertia (9.206 kg — Skandha's own real torso-shell value, not
a substitution) is kept **unchanged**, even though it no longer has arms/head as
children. This is a deliberate simplification: no attempt was made to guess how much
additional ballast mass a real physical lower-body-only rig would need on the torso
to represent the removed arms/head. **If the real test rig keeps the arms/head
physically attached as dead weight (not actuated) rather than removing them
entirely, `base_link`'s mass/inertia here will need to be increased accordingly —
this is a question for the robotics team, not something resolved here.**

## 1. How the XML was derived

`src/assets/robots/skandhalowerbody/xmls/SkandhaLowerBody.xml` was generated
programmatically from `src/assets/robots/skandharobot/xmls/SkandhaRobot.xml` (not
hand-edited) by:

1. Deleting the `RA_link_1`, `LA_link_1`, and `head_link_1` body subtrees (arms,
   head — deleting the top body of each subtree removes all of its children too),
   and the three chest-camera dummy bodies (`lower_chest_camera`,
   `upper_chest_front_camera`, `upper_chest_rear_camera`).
2. Pruning `<asset><mesh>` entries down to only the meshes still referenced by a
   remaining `<geom>` (48 of the original meshes survive; the arm/head/hand/camera
   meshes do not).
3. Pruning `<sensor>` entries that referenced a removed joint (all `jointpos` /
   `jointvel` / `jointactuatorfrc` sensors for the 17 removed joints — 51 sensor
   entries dropped). The base `imu` site sensors and the hip-mounted
   `vectornav_hip_front_site_imu` sensors are untouched (their sites are on bodies
   that still exist).
4. Pruning `<contact><exclude>` pairs that referenced a removed body (6 of the
   original 10 self-collision excludes were arm/head-specific and dropped; the 4
   leg/hip excludes remain).
5. Copying only the 48 surviving mesh `.STL` files (visual + collision-hull pairs)
   into this robot's own `xmls/assets/` — this robot does **not** read from
   `skandharobot`'s asset folder at runtime; it is self-contained, matching every
   other robot folder in this repo.

This is a mechanical/structural trim only. **No joint range, effort limit, mass, or
PD gain of any surviving joint/link was changed from `skandharobot`'s values** — see
`skandha_robot_constants.py` / `skandharobot/README.md` for their derivation and the
caveats already documented there (e.g. ankle_roll's ±20° fallback range, the
mass-substitution rationale) — all of those caveats still apply here unchanged, since
none of the surviving hardware was touched.

## 2. Home keyframe

Reuses `skandharobot`'s exact home-keyframe base height and leg/waist joint angles
(no re-derivation needed — leg kinematics are unaffected by removing the arms/head).
Verified directly on this trimmed model: with these same values, all 8
foot-contact-sphere geoms land within **<1 mm** of world Z = 0.

```python
pos = (0, 0, 0.8692443230706399)
hip_pitch_r=+0.2057   hip_pitch_l=-0.2057
knee_pitch_r=-0.1822  knee_pitch_l=+0.1822
ankle_pitch_r=-0.0849 ankle_pitch_l=+0.0849
# everything else = 0 (no arm joints exist to set)
```

## 3. Actuators

Identical gains/limits to `skandharobot`'s corresponding joint groups (same
hardware) — just the leg + waist groups; the arm/head/wrist actuator groups are not
instantiated at all.

| Joint group | stiffness | damping | effort limit (N·m) | armature |
|---|---|---|---|---|
| hip_roll | 180 | 6.0 | 235 | 0.01 |
| hip_yaw | 180 | 6.0 | 235 | 0.01 |
| hip_pitch | 220 | 8.0 | 330 | 0.01 |
| knee_pitch | 250 | 8.0 | 330 | 0.01 |
| ankle_pitch | 60 | 2.5 | 55 | 0.01 |
| ankle_roll | 60 | 2.5 | 55 | 0.01 |
| waist | 150 | 5.0 | 91 | 0.01 |

`SKANDHA_LOWER_BODY_ACTION_SCALE` is `0.25 * effort_limit / stiffness` per joint,
same formula as `skandharobot`.

## 4. Registered tasks

```
SkandhaLowerBody-Rough   # generated rough terrain, terrain curriculum, height scan
SkandhaLowerBody-Flat    # flat plane, no height scan
```

Same reward/observation/event wiring as `Skandha-*` (see
`src/tasks/velocity/config/skandhalowerbody/env_cfgs.py`), with the arm/head entries
dropped from the `pose` reward's per-joint-category standard deviations (there are no
arm/head joints to shape).

Observation dimensions: actor **50** / critic **65** (flat terrain) — vs. full
Skandha's 101 / 116, the difference being the 17 fewer joints (34 fewer
position+velocity+action values) and 3 fewer command-adjacent... actually just the
joint-count-driven terms shrinking (`joint_pos`/`joint_vel`/`actions`: 30→13 each).
Action dimension: **13**.

## 5. Training

```bash
# Flat terrain (start here)
python scripts/train.py SkandhaLowerBody-Flat --env.scene.num-envs=4096

# Rough terrain (after flat-ground walking is solid)
python scripts/train.py SkandhaLowerBody-Rough --env.scene.num-envs=4096
```

Run from the `unitree_rl_mjlab/` directory with `PYTHONPATH` including that directory
(e.g. `PYTHONPATH=. python scripts/train.py ...` on Windows/PowerShell:
`$env:PYTHONPATH="."`). Logs/checkpoints land in
`logs/rsl_rl/skandhalowerbody_velocity/<date_time>/`.

## 6. Verified

| Check | Result |
|---|---|
| MJCF compiles natively in MuJoCo (`mujoco.MjSpec.from_file` → `.compile()`) | PASS |
| `mj_forward` at the home keyframe, no NaNs | PASS |
| Home-keyframe flat-footedness (reused values, re-verified on this model) | PASS — all 8 foot-contact spheres within <1 mm of Z=0 |
| Task registration (`SkandhaLowerBody-Flat` / `-Rough` appear in `scripts/list_envs.py`) | PASS |
| Env build + observation/action wiring (`ManagerBasedRlEnv`) — actor 50-dim (flat) / 237-dim (rough), action 13-dim | PASS |
| Short rollout, flat: 50-step zero-action + 100-step random-action, 4 parallel envs, no NaNs | PASS |
| Short rollout, rough: 50-step zero-action + 100-step random-action, 4 parallel envs, no NaNs | PASS |
| `Skandha-Flat` regression (full-body robot still builds/steps unmodified after this addition) | PASS |

Not yet run: an actual PPO training session (the parent `skandharobot` integration
verified 3 PPO iterations end-to-end with an ONNX export; the same is recommended
here before large-scale training investment, but was not run as part of this
integration).
