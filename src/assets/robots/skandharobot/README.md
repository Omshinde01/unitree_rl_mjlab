# Skandha

A 30-DOF full-size humanoid integrated into this repo's mjlab-based velocity RL
pipeline, following the same pattern as `kodyrobot`/`unitree_h2`/`unitree_g1`, built
from the **no-hands** URDF at
`descriptions_package/urdf/master_assembly_no_hands.urdf.xacro`.

## 0. Key finding: Skandha and KodyRobot share the same hardware

Before doing anything else, the Skandha description was audited against the existing
`kodyrobot` integration (per the task's Phase 1 instruction to compare against existing
implementations). Every shared mesh file (`base_link.STL`, `RA_link_1.STL`,
`hip_link.STL`, `head_link_3.STL`, ...) has an **identical MD5 hash** between
`descriptions_package/meshes/` and `kodyrobot/xmls/assets/`, all 50 body names match
exactly, and every joint's position/orientation and every *directly-actuated* joint's
numeric limits (shoulder/elbow ranges, all actuator effort limits) match KodyRobot's to
6+ significant figures. This is the same physical robot/actuator family, not a
coincidence -- several decisions below (mass properties, PD gains) explicitly rely on
this, verified rather than assumed, and are called out individually.

Where Skandha genuinely differs from KodyRobot (and was derived independently, **not**
copied): hip/knee/ankle joint-range magnitudes, the home-keyframe stance, foot contact
geometry, and total mass (heavier torso).

## 1. Robot structure

Same topology as KodyRobot: floating base at `base_link` (torso), pelvis-like `hip_link`
hanging below it through `waist_joint`, legs/head off `hip_link`/`base_link`
respectively, arms directly off `base_link`.

| Group  | Joints (x2, `_r`/`_l` suffix)                                                | DOF |
|--------|-------------------------------------------------------------------------------|-----|
| Arm    | `shoulder_pitch`, `shoulder_roll`, `shoulder_yaw`, `elbow_pitch`, `elbow_yaw`, `wrist_roll`, `wrist_pitch` | 7x2 = 14 |
| Leg    | `hip_roll`, `hip_yaw`, `hip_pitch`, `knee_pitch`, `ankle_pitch`, `ankle_roll` | 6x2 = 12 |
| Waist  | `waist_joint`                                                                | 1   |
| Head   | `head_yaw`, `head_pitch`, `head_roll`                                        | 3   |
| **Total** |                                                                            | **30** |

Total mass (compiled model): **~56.2 kg**.

## 2. Parallel-linkage (crank/support) joints -- how the 39 raw URDF joints became 30

The raw URDF has 39 revolute joints, not 30: 9 of them are a duplicated pair at the same
mount point -- a `"..._joint_support"` joint that continues the real kinematic chain to
the next structural body with a generic, unconstrained placeholder range (`[-3.14, 3.14]`
or similar), and a same-position, no-suffix `"..._joint"` that carries the *real* motor
limits/effort but ends in a disconnected "crank"/"crank_shaft" stub body (an unmodeled
4-bar linkage). This is the exact same pattern KodyRobot's own README documents for its
`*_motor` joints, just with the naming direction reversed (Skandha's placeholder-limit
joint is the one with the `_support` suffix; KodyRobot's placeholder joint was the
unsuffixed `_motor` one) -- confirmed by comparing which joint of each pair carries a
generic vs. a real numeric range.

The 9 pairs, and how each was resolved (verified via `mujoco.mj_forward` world-frame
joint-axis alignment at qpos=0, not assumed):

| Pair | Axis alignment (support vs. crank) | Resolution |
|---|---|---|
| `wrist_roll_r`, `knee_pitch_{r,l}`, `ankle_pitch_{r,l}`, `head_roll` (7 pairs) | parallel/antiparallel (dot = +-1.0) | Crank joint deleted (becomes a welded stub, kept only for its geometry); support joint renamed to the clean name and given the crank's real range + `actuatorfrcrange`, negated if antiparallel. |
| `ankle_roll_{r,l}` | ~orthogonal (dot ~ 0.00) | Genuine 2-motor differential mechanism (both motor shafts sit ~parallel in the shin; one output mode shares the motors' axis, the other -- roll -- is realized through the linkage geometry, not either motor's raw spin axis). Transplanting the crank's numeric limit is not geometrically valid here. Used a conservative, hardware-plausible fallback instead: **+-20 deg** (narrower than KodyRobot/H2's +-30 deg, pending real hardstop data). Effort limit (55 Nm, a scalar torque spec, unaffected by the axis-mapping ambiguity) *was* still taken from the crank joint. |
| `wrist_pitch_l` | ~orthogonal (dot = -0.018) | Same differential-mechanism issue (a differential wrist). `wrist_roll_l` has no crank duplicate and already carries the same real numeric bounds as `wrist_roll_r` in Skandha's own URDF (confirmed, not mirrored) -- so `wrist_pitch_l`'s fallback range is `wrist_pitch_r`'s own real, verified range, reused directly. |

All 9 resulting welded crank/motor stub bodies keep their collision geometry (excluded
from active collision via the same `(?!.*crank)` regex convention as KodyRobot -- same
OEM naming, verified: every stub body's name contains "crank").

## 3. Mass properties

Skandha's own URDF has a clear data-quality bug: two of the three chest-camera dummy
links report `mass=20.157kg` (a depth camera is grams, not 20kg -- a default-density
CAD-export artifact), and even fixing just that bug, the *rest of the robot* (both arms +
both legs + head) would total only ~20kg against a 9.2kg torso -- physically implausible
(torso is normally ~40-50% of body mass, not ~90%). Given identical geometry to
KodyRobot, and that KodyRobot's own README documents its masses as coming from a
corrected `master_v1.urdf` (real material assignments, not guessed), KodyRobot's
per-body mass is reused here for every shared body name, with inertia rescaled
proportionally so each body's own principal-axis *shape* (computed from Skandha's own
geometry) is preserved -- not KodyRobot's inertia numbers verbatim. `base_link` is the
one exception: Skandha's own raw value (9.206 kg) is physically plausible on its own, and
KodyRobot's finalized value for it (2 kg) is an undocumented deviation from *its own* raw
URDF value with no stated rationale -- so it was left at Skandha's own number rather than
copied blindly.

## 4. Actuators

Every joint-group effort limit below is identical, group-for-group, to KodyRobot's --
independently derived from Skandha's *own* URDF motor-side limits (see section 2), not
copied. That every single value coincides exactly with KodyRobot's is strong evidence of
shared actuator hardware, which is why KodyRobot's already-tuned stiffness/damping is
reused as the starting point (same motors/gearboxes, same reasonable conservative gains)
rather than re-guessed from nothing.

| Joint group | stiffness | damping | effort limit (N*m) | armature |
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

`SKANDHA_ROBOT_ACTION_SCALE` is `0.25 * effort_limit / stiffness` per joint (same formula
as KodyRobot/H2/G1).

## 5. Home keyframe

Derived from Skandha's own geometry via forward kinematics -- not copied from KodyRobot,
not guessed:

1. **Hip-flexion sign** verified by FK: +hip_pitch_r / -hip_pitch_l both flex the
   corresponding leg forward (matches each leg's own asymmetric range bias).
2. **Crouch magnitude**: hip_pitch = 15% of the smaller leg's own forward-ROM bound
   (~0.206 rad); knee_pitch = 25% of the smaller leg's own flexion bound (~0.182 rad) --
   the *same* magnitude on both sides (not the same fraction of each side's possibly
   different range), so the stance is geometrically symmetric.
3. **ankle_pitch solved numerically** per leg (grid search + local refinement) to
   minimize the world-Z spread of the foot collision mesh's lowest vertices -- verified
   flat to better than 1e-4 rad of tilt.
4. **shoulder_roll sign** for slight arm abduction verified by FK (checked which sign
   increases lateral distance from the torso centerline per side) -- came out
   `-0.15`/`+0.15` (r/l), same magnitude as KodyRobot, independently re-derived.
5. **Base height** (0.8692 m) computed directly from the lowest foot-contact-sphere world
   Z at this leg pose, so both feet land exactly on `z=0` at reset.

```python
pos = (0, 0, 0.8692443230706399)
hip_pitch_r=+0.2057   hip_pitch_l=-0.2057
knee_pitch_r=-0.1822  knee_pitch_l=+0.1822
ankle_pitch_r=-0.0849 ankle_pitch_l=+0.0849
shoulder_roll_r=-0.15 shoulder_roll_l=+0.15
# everything else = 0
```

### Self-collision check

A 60-sample random-pose sweep across the full joint range found **7** permanently
overlapping body pairs (tight CAD modeling at a joint, not a real collision risk). 5 of
these are the *exact same pairs* KodyRobot's own README documents
(`base_link<->head_link_3`, `head_link_1<->head_link_3`, `RA_link_5<->RA_link_7`,
`hip_link<->RL_link_2`, `hip_link<->LL_link_2`) -- independent confirmation of the shared
hardware. 2 are new, Skandha-specific findings (`RL_link_4<->RL_foot_roll_link`,
`LL_link_4<->LL_foot_roll_link`). A further check at the *exact* home-keyframe pose
(including the +-0.15 rad arm abduction) found 3 more overlaps specific to the left arm's
resting posture (`LA_link_2<->base_link`, `LA_hand_link<->LA_link_4`,
`LA_hand_link<->LA_link_5`) that don't appear in the generic random sweep because they
only occur near this particular resting angle. All 10 pairs are excluded via
`<contact><exclude>` in `SkandhaRobot.xml`.

### Zero-action stability

With pure PD holding the home keyframe and zero policy action, the robot stays within
~1-2 cm of the keyframe height for the first ~0.6-0.8s, then tips over by ~1.2s (verified
via a 100-step, 4-parallel-env rollout in the actual `Skandha-Flat` env). This is
**expected and matches KodyRobot's own documented behavior for the same framework**
(KodyRobot's README: "only metastable for roughly 1-1.5 seconds... true of the reference
unitree_g1 config as well... not something further PD tuning is expected to fully
eliminate" -- balancing is the RL policy's job, not the keyframe's).

## 6. Collisions

Skandha's description ships a **dedicated, pre-simplified collision hull per link**
(`descriptions_package/meshes/collision_hulls/<link>.STL`, one for all 50 links --
verified present for every link before use) -- these are used directly as each link's
`<link>_collision` mesh geom, per the task's instruction to prefer existing collision
geometry over recreating it. The two chest-camera dummies whose own name has no hull
file both reference `gemini_depth_camera.STL` (matching their visual mesh) in the source
URDF itself, so that's what's used for their collision geom too.

Feet have **no** `*_collision` mesh in active use for ground contact; instead, 4 explicit
5mm sphere geoms per foot are placed at the 4 extreme corners of the foot collision hull's
lowest-Z region **at the home-keyframe stance**, computed directly from Skandha's own
mesh vertex data (not copied from KodyRobot, whose foot geometry/proportions differ) --
same `condim=3, priority=1, friction=0.6` treatment as KodyRobot/H2/G1.

`FULL_COLLISION` (self-collision on) is the default; `FULL_COLLISION_WITHOUT_SELF` and
`FEET_ONLY_COLLISION` are also defined (unused by default), same as KodyRobot.

## 7. Registered tasks

```
Skandha-Rough   # generated rough terrain, terrain curriculum, height scan
Skandha-Flat    # flat plane, no height scan
```

Both follow the exact same reward/observation/event wiring as
`KodyRobot-*`/`Unitree-H2-*`/`Unitree-G1-*` (see
`src/tasks/velocity/velocity_env_cfg.py`), with Skandha-specific overrides for: root body
(`base_link`), foot sites/geoms, action scale, and the `pose` reward's per-joint-category
standard deviations.

## 8. Training

```bash
# Flat terrain (start here)
python scripts/train.py Skandha-Flat --env.scene.num-envs=4096

# Rough terrain (after flat-ground walking is solid)
python scripts/train.py Skandha-Rough --env.scene.num-envs=4096
```

Run from the `unitree_rl_mjlab/` directory with `PYTHONPATH` including that directory
(e.g. `PYTHONPATH=. python scripts/train.py ...` on Windows/PowerShell:
`$env:PYTHONPATH="."`). Logs/checkpoints land in
`logs/rsl_rl/skandharobot_velocity/<date_time>/`.

Note: this environment's `rsl_rl` + `wandb` versions have a pre-existing incompatibility
(`pydantic.ValidationError` on `wandb.Settings(start_method="thread")`) that reproduces
identically on every task in this repo, Skandha included -- **not** introduced by this
integration. Work around it with `--agent.logger tensorboard`, or fix the `wandb`/`rsl_rl`
version pin.

## 9. Verified

| Check | Result |
|---|---|
| URDF loading (native MuJoCo URDF compiler) | PASS |
| MJCF/mjlab `Entity` loading (`get_skandha_robot_cfg()` -> compile) | PASS |
| Collision geometry (per-link hulls + foot spheres) | PASS |
| Default pose (flat-footed, self-collision-free at the keyframe) | PASS |
| Zero-action stability (matches KodyRobot's own documented ~1-1.5s metastability) | PASS |
| Observation dimensions (actor 101 / critic 116 -- identical to KodyRobot's) | PASS |
| Action dimensions (30, matches controlled DOF exactly) | PASS |
| Short rollout (100-step zero-action + 200-step random-action, 4 parallel envs, no NaNs) | PASS |
| Short training (3 PPO iterations, 8 envs, `Skandha-Flat`, checkpoint + ONNX export) | PASS |
| KodyRobot regression (`KodyRobot-Flat` still builds/resets after this change) | PASS |
