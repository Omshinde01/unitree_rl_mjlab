# Skandha — Robotics Team Requirements for Sim-to-Real RL Deployment

**Audience:** Robotics / hardware / firmware team
**Author context:** Derived from the current mjlab-based RL integration of Skandha
(`src/assets/robots/skandharobot/`, `src/tasks/velocity/config/skandharobot/`) — a
30-DOF full-size humanoid trained with the `Skandha-Flat` / `Skandha-Rough` velocity
tracking tasks (PPO, `rsl_rl`).
**Purpose:** List every physical parameter, sensor, and interface the training team
needs from the robotics team for the simulated policy to transfer safely and
predictably onto the real robot, and specify exactly what the trained policy will
expect as input and produce as output at deployment time.

---

## 0. TL;DR — what to hand back

| # | Item | Priority |
|---|------|----------|
| 1 | Signed-off per-joint position/torque limits (Section 2.1) | **Compulsory** |
| 2 | Real per-link mass/inertia/CoM, or confirmation the current values are acceptable (Section 2.2) | **Compulsory** |
| 3 | Confirmed IMU mounting location + axis orientation (Section 2.4) | **Compulsory** |
| 4 | Confirmation that joints accept an external position target at ≥50 Hz with an onboard PD/impedance loop at ≥200 Hz, or the actual achievable rates (Section 3) | **Compulsory** |
| 5 | Resolution of the 3 flagged kinematic ambiguities — ankle_roll x2, wrist_pitch_l (Section 2.6) | **Compulsory** |
| 6 | Foot sole geometry + real ground-contact points (Section 2.3) | **Compulsory** |
| 7 | Encoder resolution/bias and joint-torque-sensing availability (Section 2.5) | Recommended |
| 8 | Real friction, backlash, gear efficiency numbers (Section 2.7) | Optional (improves DR realism) |
| 9 | E-stop / safety-cutoff behavior and comms link spec (Section 5) | **Compulsory** |

---

## 1. Robot Overview (as currently modeled in simulation)

- **DOF:** 30 actuated joints — 2×7-DOF arms, 2×6-DOF legs, 1-DOF waist, 3-DOF head.
- **Total mass (sim):** ~56.2 kg.
- **Floating base:** `base_link` (torso/chest). The pelvis-like `hip_link` hangs
  *below* the torso through a single `waist_joint` — i.e. the waist is a torso-to-pelvis
  joint, not a torso-to-ground joint. Legs mount on `hip_link`; arms and head mount
  directly on `base_link`.
- **Source of truth for current sim values:** `src/assets/robots/skandharobot/skandha_robot_constants.py`
  and `src/assets/robots/skandharobot/README.md`. Everything in this document is
  derived from those files plus `SkandhaRobot.xml`; **all numeric values below are
  simulation defaults, several explicitly flagged in the source README as
  derived/assumed rather than measured, and need robotics-team verification.**

---

## 2. Parameters Required From the Robotics Team

### 2.1 Joint limits and actuator torque limits (Compulsory)

The table below is what the simulation currently uses. Please confirm/correct each
row against the actual hardware datasheet or measured hardstops — **training on
wrong limits either wastes policy capacity (limits too tight) or produces motion the
real joint cannot execute / will fault on (limits too loose).**

| Joint (×2 for `_r`/`_l` unless noted) | Range (rad) | Range (deg) | Effort limit (N·m) | Sim stiffness (Kp) | Sim damping (Kd) |
|---|---|---|---|---|---|
| hip_roll | −0.82 / +0.71 (asymmetric L vs R, see raw CSV) | ≈ −47° / +41° | 235 | 180 | 6.0 |
| hip_yaw | −1.73 / +1.42 | ≈ −99° / +82° | 235 | 180 | 6.0 |
| hip_pitch | −1.37 / +1.39 | ≈ −79° / +79° | 330 | 220 | 8.0 |
| knee_pitch | −1.07 / +0.73 | ≈ −61° / +42° | 330 | 250 | 8.0 |
| ankle_pitch | −0.65 / +0.65 | ≈ −37° / +37° | 55 | 60 | 2.5 |
| ankle_roll | **±0.349 (±20°) — fallback value, not from real hardstops** | ±20° | 55 | 60 | 2.5 |
| waist_joint (×1) | −2.97 / +3.14 | ≈ −170° / +180° | 91 | 150 | 5.0 |
| shoulder_pitch | −2.97 / +2.97 | ≈ ±170° | 91 | 90 | 4.0 |
| shoulder_roll | asymmetric per side (−2.6/+0.26 vs −0.26/+2.6) | ≈ 150°/15° | 95 | 85 | 4.0 |
| shoulder_yaw | −2.97 / +2.97 | ≈ ±170° | 35 | 65 | 3.0 |
| elbow_pitch | −2.62 / +2.62 | ≈ ±150° | 35 | 60 | 3.0 |
| elbow_yaw | −2.97 / +2.97 | ≈ ±170° | 24 | 45 | 2.5 |
| wrist_roll | −1.31 / +1.66 | ≈ −75°/+95° | 6.3 | 20 | 1.0 |
| wrist_pitch_r | −0.785 / +1.047 | ≈ −45°/+60° | 6.3 | 20 | 1.0 |
| wrist_pitch_l | **= wrist_pitch_r's range, reused directly — not independently derived** | same as above | 6.3 | 20 | 1.0 |
| head_yaw (×1) | −1.57 / +1.57 | ±90° | 6.3 | 20 | 1.0 |
| head_pitch (×1) | −0.44 / +0.44 | ±25° | 6.3 | 20 | 1.0 |
| head_roll (×1) | −0.45 / +0.45 | ±26° | 6.3 | 15 | 0.8 |

Full per-joint numeric detail (both sides, exact values): `mujoco_parsed/joints.csv`
in this repo.

**Please confirm or correct:**
- Every range above against real hardstops (soft *and* hard limits — training uses
  `soft_joint_pos_limit_factor = 0.9`, i.e. the policy is trained to treat 90% of the
  listed range as the usable range; the remaining 10% is a safety margin before the
  mechanical hardstop).
- Every effort/torque limit (continuous **and** peak/short-duration rating — the
  simulation only encodes one number per joint group).
- **Stiffness/damping (Kp/Kd) are not hardware specs** — they are simulated PD gains
  used by MuJoCo's internal position servo and are currently *reused from a sibling
  robot (KodyRobot) on the assumption of shared actuator hardware* (see Section 2.6).
  These are a training-time approximation only. The robotics team must supply the
  actual achievable Kp/Kd range (or native impedance-control range) of each joint's
  motor controller so the deployed policy's implicit gain assumption can be matched
  or the policy retrained with the real values.

### 2.2 Mass, inertia, and center of mass (Compulsory)

Skandha's own CAD/URDF export has a known data-quality bug (two chest-camera dummy
links reporting 20.157 kg — clearly a default-density artifact for what should be a
gram-scale depth camera). With that bug alone removed, the rest of the robot
(arms+legs+head) totals only ~20 kg against a 9.2 kg torso, which is not physically
plausible for a humanoid of this size. **The simulation currently substitutes
KodyRobot's per-body mass/inertia** (same CAD geometry, verified by identical mesh
hashes) for every link **except** `base_link`, which keeps Skandha's own raw URDF
value (9.206 kg). Resulting total: **~56.2 kg**.

**Required from robotics team:**
- Actual measured (or at minimum, correctly-specified CAD) mass for every link,
  especially: `base_link` (torso, currently 9.206 kg — please confirm this is
  correct, since it drives ~15% of total robot mass and the whole-body CoM), both
  chest camera dummy bodies, and battery/compute-carrying links.
  battery/compute-carrying links.
- Total robot mass (fully cabled/battery-loaded, as it will be commissioned) —
  needed to sanity-check the ~56.2 kg simulation figure before large training runs.
- CoM location per major link if available (used for domain-randomization ranges,
  Section 2.7).

### 2.3 Foot / ground-contact geometry (Compulsory)

The simulation does **not** use a full collision mesh for feet. Ground contact is
modeled as 4 explicit 5 mm point-contact spheres per foot, placed at the extreme
low-Z corners of each foot's collision hull *at the home-keyframe stance*. This
directly determines the support polygon the policy is trained to balance/walk over.

**Required:**
- Actual sole geometry / contact-patch layout (flat plate? individually sprung
  toe+heel pads? pressure-sensor locations?) if it differs from "4 rigid points near
  the foot corners."
- Foot coefficient of friction against the target deployment surface(s) — sim
  currently randomizes foot-geom friction uniformly in **[0.3, 1.6]** at env startup
  (`foot_friction` domain-randomization event); please confirm this bracket is a
  reasonable span for the real floor materials the robot will walk on.
- Confirmation of which physical structure the two named foot sites correspond to —
  `left_foot` / `right_foot` — sim uses these both for reward shaping (foot
  clearance/slip) and as the observation source for foot height.

### 2.4 IMU: mounting location and specification (Compulsory — currently ambiguous)

The XML ships **two separate IMU-like sensor blocks**, and it is not resolved which
one corresponds to the real onboard IMU used at deployment:

1. **`imu` site** — located at `base_link` (torso) origin. This is the site the
   *trained policy's observations actually read* (`imu_ang_vel` → gyro,
   `imu_lin_vel` → velocimeter, plus `projected_gravity` which is derived from the
   simulated torso orientation, not a raw sensor).
2. **`vectornav_hip_front_site_imu_site`** — located at the *hip* (`hip_link`),
   carrying a full IMU sensor suite (`framequat`, `gyro`, `accelerometer`, `framepos`,
   `framelinvel`) that is **not currently wired into the RL observation space** but
   whose naming ("vectornav") strongly implies it is modeling a real, named IMU part
   (VectorNav) that robotics has actually mounted on the hip.

**Required — this must be resolved before deployment, not after:**
- Where is the *actual* IMU physically mounted on the robot — torso (`base_link`) or
  hip (`hip_link`)? If hip-mounted, the policy's angular-velocity and
  gravity-projection observations must either be transformed into the torso frame at
  runtime (needs the fixed torso↔hip transform, i.e. the waist joint angle at all
  times) or the policy must be retrained with hip-frame IMU observations to match
  reality.
- Exact IMU model/datasheet: gyro noise density, accelerometer noise density,
  sampling rate, and whether it ships onboard orientation/attitude estimation
  (quaternion output) or only raw gyro+accel that the deploy stack must fuse.
- Mounting orientation (axis alignment) relative to the body frame used in the XML,
  so any offset rotation can be corrected in software.

### 2.5 Encoders and joint sensing (Recommended)

- Joint encoder resolution/quantization per joint (affects how cleanly `joint_pos`
  observations can match sim, which assumes continuous double-precision values).
- Whether joint-level torque/current sensing is available. The XML defines
  `jointactuatorfrc` sensors for all 30 joints, but **the current reward/observation
  config does not consume them** — if real torque feedback exists and is desired for
  future reward shaping or safety monitoring, flag it now so it can be added to the
  observation space intentionally (adding it later requires retraining).
- Zero-position calibration procedure: sim applies a per-joint random "encoder bias"
  in **±0.015 rad** during domain randomization (`encoder_bias` event) to make the
  policy robust to small joint-zero miscalibration. Please confirm this is a
  reasonable bound for your homing/calibration procedure's expected residual error —
  if real miscalibration can exceed this, the policy may not be robust to it.

### 2.6 Kinematic ambiguities that must be resolved (Compulsory)

Skandha's raw URDF represents several joints as duplicated "support + crank" pairs
(an artifact of an unmodeled 4-bar parallel linkage). For 7 of 9 such pairs, the
crank/support axes are parallel and the simplification is exact. For **3 joints**,
the mechanism is a genuine differential linkage, and the current values are
best-effort fallbacks, **not derived from real geometry**:

| Joint | Current sim assumption | What's needed |
|---|---|---|
| `ankle_roll_r` | ±20° range (conservative guess, narrower than sibling robots' ±30°), 55 N·m | Real hardstop angle range for ankle roll on both feet, and confirmation of the differential-linkage transmission ratio between the two ankle motors and roll/pitch output |
| `ankle_roll_l` | same as above | same as above |
| `wrist_pitch_l` | reused `wrist_pitch_r`'s range verbatim (unmirrored) | Real hardstop range for the *left* wrist pitch specifically — do not assume L/R symmetry here, since this assembly's own data shows some values do *not* mirror |

If actual hardware ranges differ meaningfully from these fallbacks, the joint-limit
reward/termination and the action scale for these 3 joints should be corrected before
(or the policy retrained after) large-scale training investment.

### 2.7 Domain-randomization realism inputs (Optional, but improves transfer)

These parameters are currently guessed/reused from a sibling robot to make the
training-time domain randomization "wide enough to be safe," not to match Skandha
specifically. If the robotics team can supply better numbers, sim-to-real transfer
quality generally improves:

- Foot-ground friction coefficient range for actual deployment floor(s) (sim: 0.3–1.6).
- Body/base CoM offset uncertainty (sim randomizes `base_link` CoM by ±5 cm per axis
  at episode start) — real CoM uncertainty (battery swap, payload variation) may be
  smaller or larger.
- External-push disturbance magnitudes used in training (velocity kicks up to
  ±0.5 m/s linear, ±0.4 m/s vertical, ±0.78 rad/s yaw every 5–6 s) — confirm these are
  representative of expected real-world disturbances (bumps, light shoves) for the
  intended deployment environment.
- Actuator armature (rotor inertia), currently a flat 0.01 for every joint — real
  per-joint-group values would be more accurate if available from motor datasheets.

---

## 3. Physical Robot → Policy: Required Inputs, Interfaces, and Timing (Compulsory)

The trained policy is a fixed-frequency, memory-less (no recurrent state) controller.
Every control tick it needs a single observation vector; every joint needs to accept
a position target from the policy and turn it into torque locally.

### 3.1 Control loop structure the policy assumes

| Stage | Rate | Who runs it |
|---|---|---|
| Policy inference (produces 30 action values) | **50 Hz** (env `decimation=4` × physics `timestep=0.005 s`) | Onboard compute (policy runtime, e.g. ONNX/LibTorch) |
| Low-level joint position/impedance control (PD loop turning "target angle" into torque) | **≥ 200 Hz assumed in sim** (physics substeps between policy ticks) | Robot's own joint/motor controllers — **must be confirmed available at this rate or the policy must be retrained at the achievable rate** |
| Episode-level bookkeeping (gait-phase clock, described below) | continuous, software-side | Deploy-stack wrapper, not the robot |

**This is the single most important interface requirement:** the robotics team must
confirm each joint controller can (a) accept an external absolute position target at
≥50 Hz over whatever comms link is used, and (b) internally close a position/impedance
loop at a rate fast enough to be well-approximated by the simulated PD gains in
Section 2.1 between policy updates. If the real controllers run torque-mode only, or
at a much lower position-target update rate, that must be flagged — it changes the
action-space design, not just a tuning parameter.

### 3.2 Policy observation vector ("actor" input) — what the robot must supply each tick

The `Skandha-Flat` policy's actor observation is a **101-dimensional** vector
(flat terrain — the `Skandha-Rough` variant additionally appends a terrain
height-scan grid, which requires exteroceptive sensing not itemized here since flat
terrain is the recommended first deployment target per the existing README). All
terms are concatenated in the fixed order below:

| # | Term | Dims | Source at deployment | Compulsory from robot? |
|---|---|---|---|---|
| 1 | Base angular velocity (body frame) | 3 | IMU gyro | **Yes** — raw gyro reading, see Section 2.4 for which physical IMU |
| 2 | Projected gravity (body frame) | 3 | Derived: gravity vector rotated into torso frame, i.e. needs a real-time orientation/attitude estimate (roll & pitch at minimum), **not** a raw sensor channel | **Yes** — robot/deploy-stack must run an attitude estimator (e.g. AHRS/Madgwick/Mahony fusing gyro+accel, or vendor-provided orientation output) if the IMU doesn't already output orientation |
| 3 | Velocity command (vx, vy, yaw-rate) | 3 | **Not from the robot** — supplied externally by the operator/joystick/high-level planner | N/A (external input, not sensed) |
| 4 | Gait-phase clock (sin, cos) | 2 | **Not from the robot** — a software timer on the deploy side: `phase = (t mod 0.6s)/0.6s`, emitted as `[sin(2π·phase), cos(2π·phase)]`, and forced to `[0, 0]` whenever `‖commanded velocity‖ < 0.1` | N/A (must be implemented identically in the deploy stack, not read from hardware) |
| 5 | Joint positions, relative to default pose | 30 | Joint encoders, minus the trained "default"/home joint angles (Section 4) | **Yes** |
| 6 | Joint velocities, relative to default | 30 | Joint encoders (finite-difference or native velocity sensing), minus default (default joint velocities are all 0) | **Yes** |
| 7 | Last action (previous policy output, raw/unscaled) | 30 | **Not from the robot** — the deploy stack's own memory of what it sent last tick | N/A (software state) |

Note the policy does **not** require base linear velocity, foot contact state, foot
contact force, or foot air time — those exist only in the *critic* observation used
during training and are not part of the deployed actor network. This means Skandha
does **not need** a working linear-velocity estimator (no GPS/vision odometry
requirement) or foot-sole force sensors for basic deployment of this policy.

At training time, noise is injected into most of these terms (uniform noise:
±0.2 rad/s on angular velocity, ±0.05 on projected gravity, ±0.01 rad on joint
position, ±1.5 rad/s on joint velocity) specifically so the policy tolerates sensor
noise/latency at this rough magnitude — real sensor noise significantly outside these
bounds should be flagged so training noise ranges can be adjusted to match.

### 3.3 Policy action vector — what the robot receives each tick (Section 4 has the full spec)

30 continuous values, one per actuated joint, emitted at 50 Hz. See Section 4.

---

## 4. Policy Output → Physical Robot: Action Sequence and Conversion (Compulsory)

### 4.1 Action → joint target conversion

For every joint, the raw policy output `a_i` (roughly in `[-1, 1]`-ish range post
training, **not hard-clamped to exactly that interval** unless `clip_actions` is set
on the RL runner) is converted to an absolute joint position target as:

```
target_angle_i = default_joint_pos_i + a_i * scale_i
```

- `default_joint_pos_i` is the **home-keyframe** joint angle (Section 5) — not zero.
- `scale_i = 0.25 * effort_limit_i / stiffness_i` (computed per joint group from
  Section 2.1's table, e.g. hip_roll: `0.25 * 235 / 180 ≈ 0.326 rad` per unit action).
- The resulting `target_angle_i` is what a **position/impedance controller** on the
  real joint should track — the policy itself never outputs torque directly.

### 4.2 Action vector ordering — read this before wiring anything up

The 30-dim action vector's index order is **determined programmatically** by the
actuator group declaration order in `skandha_robot_constants.py`
(`SKANDHA_ROBOT_ARTICULATION.actuators`), grouped as: hip_roll → hip_yaw → hip_pitch →
knee_pitch → ankle_pitch → ankle_roll → waist → shoulder_pitch → shoulder_roll →
shoulder_yaw → elbow_pitch → elbow_yaw → wrist(roll then pitch) → head_yaw/pitch →
head_roll, with right (`_r`) before left (`_l`) within each group. The joint-position
and joint-velocity **observation** slices (Section 3.2, items 5–6) use this exact
same order.

**Do not hand-hardcode this order on the robot side.** It is an implementation detail
of how the simulation resolves regex-matched actuator names against the compiled
model's internal joint list, and is not guaranteed stable across model rebuilds. The
training team will export an authoritative `joint_names` list (ordered, one name per
action index, e.g. `["hip_roll_r_joint", "hip_roll_l_joint", ...]`) alongside every
trained checkpoint/ONNX export. **The robotics team's integration must map by joint
name against that shipped list, not by assumed index position.** This is a
compulsory deliverable from the training side and a compulsory integration
requirement on the robotics side (map-by-name, never by position).

### 4.3 Action limits / safety

- Action-rate is penalized during training (`action_rate_l2`) and joint accel is
  penalized (`joint_acc_l2`), so outputs should already be smooth — but the real
  controller should still enforce hard position/velocity/torque clamps at the values
  in Section 2.1 independent of what the policy sends, as a hardware safety backstop.
- The policy is trained with an explicit **joint-limit violation penalty**
  (`joint_pos_limits`, large negative weight) and a **fall/termination penalty**
  (`is_terminated`, weight −200) using a 70° body-tilt-from-vertical cutoff as the
  "fallen over" condition — but these are training-time shaping signals, not runtime
  safety guarantees. Physical joint limit switches / torque fuses / an emergency stop
  must exist independently of the policy (see Section 5).

---

## 5. Electrical / Comms / Safety Requirements (Compulsory)

- **Command link latency + jitter**: policy runs at 50 Hz (20 ms period); please
  specify the achievable round-trip latency from "policy computes 30 floats" to
  "joint torque changes," and its jitter, so the training team can assess whether
  additional latency-robustness domain randomization is needed (not currently
  modeled explicitly in this task).
- **E-stop behavior**: what happens to joint controllers on E-stop/comms-loss —
  torque-off (limp) vs. hold-last-position vs. controlled crouch? This affects what
  "safe" means operationally and should be decided independent of the RL policy.
  Sim/pure-PD-only rollouts show Skandha becomes unstable and falls after **~1.2 s**
  once the policy stops correcting (matches sibling-robot behavior in this repo) —
  i.e. **do not rely on the passive/PD-only pose for more than ~1 second of
  self-stability**.
- **Power/compute budget** for running the policy network onboard (actor MLP:
  3 hidden layers, 512→256→128 units, ELU activation — small enough for effectively
  any onboard CPU/GPU, but confirm the deploy target).

---

## 6. Home / Default Pose (for reference — needed to interpret Section 4.1)

The policy's joint-position observations and action targets are both relative to a
fixed **home keyframe** (a slight, flat-footed crouch), not to each joint's zero
position:

```
base height:        0.8692 m
hip_pitch_r  = +0.2057 rad   hip_pitch_l  = -0.2057 rad
knee_pitch_r = -0.1822 rad   knee_pitch_l = +0.1822 rad
ankle_pitch_r= -0.0849 rad   ankle_pitch_l= +0.0849 rad
shoulder_roll_r = -0.15 rad  shoulder_roll_l = +0.15 rad
all other joints = 0.0 rad
```

The robotics team should confirm this pose is physically reachable and
self-collision-free on the real hardware (it was only verified against the CAD/mesh
model in simulation) before using it as a startup/homing pose.

---

## 7. Open Questions Needing a Robotics-Team Answer Before Real-Robot Trials

1. Is the physical IMU on the torso or the hip? (Section 2.4 — blocks correct
   observation wiring.)
2. What are the real ankle-roll hardstops and the actual differential-linkage ratio
   for both ankles? (Section 2.6.)
3. What is the real left-wrist-pitch range? (Section 2.6.)
4. Can each joint controller accept an external position target at 50 Hz with an
   internal loop at ≥200 Hz? If not, what rates are actually achievable? (Section 3.1.)
5. What is the true `base_link` (torso) mass, and total robot mass fully loaded?
   (Section 2.2.)
6. Confirm foot contact geometry matches "4 rigid point contacts near the foot
   corners," or supply the real geometry. (Section 2.3.)
7. E-stop / comms-loss behavior for the joint controllers. (Section 5.)
