# SkandhaLowerBody

A 12-DOF lower-body-only derivative of `skandharobot`, integrated into this repo's
mjlab-based velocity RL pipeline the same way `skandharobot` itself is.

## 0. What "lower body" means here

Confirmed against the real hardware description (`lower_body.urdf.xacro` in the
standalone lower-body URDF package): the physical "hip down" test rig mounts
`hip_link` directly on a fixed bench gantry (a physical test stand — not simulated
here), with **no torso body and no waist joint above it**. The waist joint only
exists as the torso↔pelvis connection in the full humanoid; it has no meaning once
the torso is removed.

Kept: `hip_link` (now the floating base directly) and both 6-DOF legs.

Removed entirely: `base_link` (the torso), `waist_joint`, both 7-DOF arms, the
3-DOF head, and the three chest-camera dummy bodies.

| Group | Joints (×2, `_r`/`_l`) | DOF |
|---|---|---|
| Leg | `hip_roll`, `hip_yaw`, `hip_pitch`, `knee_pitch`, `ankle_pitch`, `ankle_roll` | 6×2 = 12 |
| **Total** | | **12** |

Compiled model total mass: **~35.9 kg**.

### Corrected from an earlier revision

This robot previously (incorrectly) kept `base_link` as the floating base with
`hip_link` hanging below it through `waist_joint`. Since that joint's actuator was
already commented out (no PD control on it), it existed in the compiled model as a
**free-swinging, undamped, unactuated hinge** between the torso and pelvis — not
just "unactuated" but a genuine uncontrolled extra DOF. This has been fixed:
`base_link` and `waist_joint` are now removed from the kinematic tree entirely, and
`hip_link` carries the free joint directly, matching the real hip-down hardware.

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

A follow-up pass (see "Corrected from an earlier revision" above) went one step
further and removed `base_link`/`waist_joint` too, after the standalone lower-body
URDF package confirmed the real hardware has no torso above `hip_link` at all —
`hip_link` now carries the free joint directly, and its mesh/collision/asset
registrations were simply moved up a nesting level (no new mesh files were needed;
everything was already present in this robot's own `xmls/assets/`).

## 2. Home keyframe

Reuses full Skandha's leg joint angles (leg kinematics are unaffected by which body
is the floating root) with the root height recomputed for `hip_link` now being the
root directly: full Skandha's base height (0.8692443230706399) plus the local
offset `hip_link` used to sit at under `base_link` (-0.0316317206795731), giving
0.8376126023910668. Verified directly on the corrected model: all 8
foot-contact-sphere geoms land within **<1 mm** of world Z = 0.

```python
pos = (0, 0, 0.8376126023910668)
hip_pitch_r=+0.2057   hip_pitch_l=-0.2057
knee_pitch_r=-0.1822  knee_pitch_l=+0.1822
ankle_pitch_r=-0.0849 ankle_pitch_l=+0.0849
# everything else = 0 (no arm/waist joints exist to set)
```

## 3. Actuators

Identical gains/limits to `skandharobot`'s corresponding leg joint groups (same
hardware); the arm/head/wrist/waist actuator groups are not instantiated at all —
there is no waist joint to actuate.

| Joint group | stiffness | damping | effort limit (N·m) | armature |
|---|---|---|---|---|
| hip_roll | 180 | 6.0 | 235 | 0.01 |
| hip_yaw | 180 | 6.0 | 235 | 0.01 |
| hip_pitch | 220 | 8.0 | 330 | 0.01 |
| knee_pitch | 250 | 8.0 | 330 | 0.01 |
| ankle_pitch | 60 | 2.5 | 55 | 0.01 |
| ankle_roll | 60 | 2.5 | 55 | 0.01 |

`SKANDHA_LOWER_BODY_ACTION_SCALE` is `0.25 * effort_limit / stiffness` per joint,
same formula as `skandharobot`.

## 4. Registered tasks

```
SkandhaLowerBody-Rough   # generated rough terrain, terrain curriculum, height scan
SkandhaLowerBody-Flat    # flat plane, no height scan
```

Same reward/observation/event wiring as `Skandha-*` (see
`src/tasks/velocity/config/skandhalowerbody/env_cfgs.py`), with the arm/head/waist
entries dropped from the `pose` reward's per-joint-category standard deviations
(there are no arm, head, or waist joints to shape).

Action dimension: **12** (legs only — no waist).

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
| Home-keyframe flat-footedness (root height recomputed for `hip_link`-as-root) | PASS — all 8 foot-contact spheres within <1 mm of Z=0 |
| `get_skandha_lower_body_cfg()` → `Entity(...).spec.compile()` — nu=12, no `base_link`/`waist_joint` | PASS |
| Task registration (`SkandhaLowerBody-Flat` / `-Rough`) | PASS |
| Env build + 300-step random-action rollout, 4 parallel envs, both terrain variants | PASS, no NaNs |
