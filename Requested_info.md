# Skandha

What we need from you, and exactly what the trained policy reads/sends each tick.

---

## 1. What we need from you

**Compulsory** = training/deployment is blocked or unsafe without it.
**Optional** = improves accuracy/realism but training can proceed with a reasonable placeholder.

### Mechanical

| # | Item                                                                                    | Priority   |
| - | --------------------------------------------------------------------------------------- | ---------- |
| 1 | Full joint list confirmation: names, DOF count, kinematic tree (parent/child per joint) | Compulsory |
| 2 | Joint position limits — soft and hard hardstops, every joint                           | Compulsory |
| 3 | Joint torque limits — continuous and peak, every joint                                 | Compulsory |
| 4 | Joint velocity limits, every joint                                                      | Compulsory |
| 5 | Per-link mass, inertia tensor, and center of mass                                       | Compulsory |
| 6 | Total robot mass, fully cabled/battery-loaded                                           | Compulsory |
| 7 | Foot sole geometry and ground-contact layout                                            | Compulsory |
| 8 | Ground-contact friction coefficient(s) for target deployment surface(s)                 | Optional   |

### Actuators

| #  | Item                                                                         | Priority   |
| -- | ---------------------------------------------------------------------------- | ---------- |
| 9  | Control modes available per joint (position / velocity / torque / impedance) | Compulsory |
| 10 | Native control-loop rate achievable per joint                                | Compulsory |
| 11 | Achievable Kp/Kd (or impedance-gain) range per joint                         | Compulsory |
| 12 | Gear ratio, transmission type, and backlash per joint                        | Optional   |
| 13 | Rotor/armature inertia per joint                                             | Optional   |

### Sensing

| #  | Item                                                                                                          | Priority   |
| -- | ------------------------------------------------------------------------------------------------------------- | ---------- |
| 14 | IMU: exact mounting location + orientation on the robot                                                       | Compulsory |
| 15 | IMU datasheet: gyro/accel noise density, sample rate, whether it outputs a fused orientation or raw data only | Compulsory |
| 16 | Joint encoder type, resolution, and update rate                                                               | Compulsory |
| 17 | Joint torque/current sensing availability, if any                                                             | Optional   |
| 18 | Foot force/pressure sensing availability, if any                                                              | Optional   |
| 19 | Any other onboard sensors relevant to balance/locomotion (e.g. additional IMUs, cameras, lidar)               | Optional   |

> Electrical/Compute/Comms and Safety items (comms protocol, latency, E-stop
> behavior, fault limits, etc.) are intentionally left out of this ask — those are
> live-hardware / bring-up concerns that only matter once the policy is actually
> commanding powered motors on the robot. Not needed for the current
> inference-pipeline / preliminary-URDF stage.

---

## 2. Model Input (what the robot must send TO the policy, every tick)

101 numbers, always in this exact order. Example row = one real tick, illustrative values.

| # | Field                                              | Dims | Index range | Where it comes from                       | Example values        |
| - | -------------------------------------------------- | ---- | ----------- | ----------------------------------------- | --------------------- |
| 1 | Base angular velocity (x,y,z)                      | 3    | 0–2        | IMU gyro                                  | `0.01, -0.02, 0.00` |
| 2 | Projected gravity (x,y,z)                          | 3    | 3–5        | Computed from orientation estimate        | `0.02, 0.01, -0.98` |
| 3 | Velocity command (vx, vy, yaw-rate)                | 3    | 6–8        | External (joystick / planner), not sensed | `0.50, 0.00, 0.00`  |
| 4 | Gait phase (sin, cos)                              | 2    | 9–10       | Software clock, 0.6 s cycle, not sensed   | `0.31, 0.95`        |
| 5 | Joint positions (30 joints, relative to home pose) | 30   | 11–40      | Joint encoders − home angle              | `0.00, 0.02, ...`   |
| 6 | Joint velocities (30 joints, relative to home)     | 30   | 41–70      | Joint encoders (velocity)                 | `0.00, 0.05, ...`   |
| 7 | Last action sent (previous tick's output, raw)     | 30   | 71–100     | Software memory (what we sent last time)  | `0.00, 0.01, ...`   |

Joint order for rows 5–7 is the **same fixed order** as the output table below.

Not needed by the robot: base linear velocity, foot force/contact sensors — the deployed policy does not use them.

---

## 3. Model Output (what the policy sends TO the robot, every tick)

30 numbers, one per joint, at 50 Hz.

**Important: the network's raw output is NOT a usable target angle.** It is a small
value roughly in `[-1, 1]`, and a fixed conversion must be applied per joint before
it goes anywhere near an actuator:

```
target_angle = home_angle + a × scale
```

`home_angle` and `scale` are fixed per-joint constants exported alongside the model
(not learned or recomputed at runtime) — this is a simple multiply + add per joint,
not a control algorithm. This conversion must run on the robot side (or wherever the
policy is hosted) before the value reaches the joint controller.

| #  | Joint                  | Home angle (rad) | Scale | Example raw action`a` | Example target angle (rad) |
| -- | ---------------------- | ---------------- | ----- | ----------------------- | -------------------------- |
| 1  | hip_roll_r_joint       | 0.000            | 0.326 | 0.10                    | 0.033                      |
| 2  | hip_roll_l_joint       | 0.000            | 0.326 | -0.05                   | -0.016                     |
| 3  | hip_yaw_r_joint        | 0.000            | 0.326 | 0.00                    | 0.000                      |
| 4  | hip_yaw_l_joint        | 0.000            | 0.326 | 0.00                    | 0.000                      |
| 5  | hip_pitch_r_joint      | 0.2057           | 0.375 | 0.20                    | 0.281                      |
| 6  | hip_pitch_l_joint      | -0.2057          | 0.375 | -0.20                   | -0.281                     |
| 7  | knee_pitch_r_joint     | -0.1822          | 0.330 | -0.30                   | -0.281                     |
| 8  | knee_pitch_l_joint     | 0.1822           | 0.330 | 0.30                    | 0.281                      |
| 9  | ankle_pitch_r_joint    | -0.0849          | 0.229 | 0.05                    | -0.073                     |
| 10 | ankle_pitch_l_joint    | 0.0849           | 0.229 | -0.05                   | 0.073                      |
| 11 | ankle_roll_r_joint     | 0.000            | 0.229 | 0.00                    | 0.000                      |
| 12 | ankle_roll_l_joint     | 0.000            | 0.229 | 0.00                    | 0.000                      |
| 13 | waist_joint            | 0.000            | 0.152 | 0.00                    | 0.000                      |
| 14 | shoulder_pitch_r_joint | 0.000            | 0.253 | 0.10                    | 0.025                      |
| 15 | shoulder_pitch_l_joint | 0.000            | 0.253 | -0.10                   | -0.025                     |
| 16 | shoulder_roll_r_joint  | -0.15            | 0.279 | 0.00                    | -0.150                     |
| 17 | shoulder_roll_l_joint  | 0.15             | 0.279 | 0.00                    | 0.150                      |
| 18 | shoulder_yaw_r_joint   | 0.000            | 0.135 | 0.00                    | 0.000                      |
| 19 | shoulder_yaw_l_joint   | 0.000            | 0.135 | 0.00                    | 0.000                      |
| 20 | elbow_pitch_r_joint    | 0.000            | 0.146 | 0.20                    | 0.029                      |
| 21 | elbow_pitch_l_joint    | 0.000            | 0.146 | 0.20                    | 0.029                      |
| 22 | elbow_yaw_r_joint      | 0.000            | 0.133 | 0.00                    | 0.000                      |
| 23 | elbow_yaw_l_joint      | 0.000            | 0.133 | 0.00                    | 0.000                      |
| 24 | wrist_roll_r_joint     | 0.000            | 0.079 | 0.00                    | 0.000                      |
| 25 | wrist_roll_l_joint     | 0.000            | 0.079 | 0.00                    | 0.000                      |
| 26 | wrist_pitch_r_joint    | 0.000            | 0.079 | 0.00                    | 0.000                      |
| 27 | wrist_pitch_l_joint    | 0.000            | 0.079 | 0.00                    | 0.000                      |
| 28 | head_yaw_joint         | 0.000            | 0.079 | 0.00                    | 0.000                      |
| 29 | head_pitch_joint       | 0.000            | 0.079 | 0.00                    | 0.000                      |
| 30 | head_roll_joint        | 0.000            | 0.105 | 0.00                    | 0.000                      |

**Do not hardcode this order on your side.** We will ship a `joint_names.json` list
with every model export — map by joint *name*, not by position, in case this order
ever changes.

Each `target_angle` should be tracked by the joint's own position/impedance
controller (PD loop), not driven open-loop.

---

## 4. Timing

| Loop                                                 | Rate                                |
| ---------------------------------------------------- | ----------------------------------- |
| Policy inference (produces the 30 outputs)           | 50 Hz                               |
| Joint-level position/PD control (tracks each target) | ≥ 200 Hz (needs your confirmation) |
