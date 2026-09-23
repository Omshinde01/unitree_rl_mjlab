# Skandha — Term Glossary (Training-Relevant Terms Only)

Plain-English explanation of the terms that actually feed the RL training
pipeline (robot model, observation space, action space). Deployment-only
terms (comms protocol, E-stop behavior, power budget, etc.) are intentionally
left out — see `robotics_handoff.md` §1 for those. Companion to
`robotics_handoff.md` — read that file for the actual request/spec, use this
file to understand the vocabulary.

---

## 1. Mechanical terms

| Term | What it means | Where it's needed | Why it matters |
|---|---|---|---|
| **Joint** | A single rotating connection between two links (e.g. the knee). Skandha has 30. | Everywhere — the whole model is built joint-by-joint | The unit everything else (limits, sensors, actions) is defined per. |
| **DOF (Degree of Freedom)** | One independent direction a joint can move in. Skandha's joints are all 1-DOF hinges, so DOF count = joint count = 30. | §1 Mechanical #1 | Confirms our joint list matches the real robot exactly — a missing/extra DOF breaks the whole action/observation mapping. |
| **Kinematic tree** | The parent→child chain of links and joints (e.g. torso → hip → knee → ankle → foot). | §1 Mechanical #1 | The simulator needs the exact same chain the real robot has, or forces/motions won't propagate the same way. |
| **Soft limit** | A joint-angle boundary the *policy* is trained to stay inside, tighter than the hardstop, as a safety margin. | §1 Mechanical #2 | Lets the robot avoid ever touching real hardstops during training/deployment. |
| **Hard limit / hardstop** | The physical mechanical stop that prevents a joint from rotating further — hitting it can damage the joint. | §1 Mechanical #2 | If we train with a wrong hard limit, the policy may command a motion the joint physically cannot make (fault/damage) or leave usable range unused. |
| **Torque limit (continuous / peak)** | Maximum force a joint motor can sustain indefinitely (continuous) vs. for a short burst (peak) without overheating/damage. | §1 Mechanical #3 | Sets how much "effort" the simulated actuator is allowed to use — wrong values mean the trained policy either can't do what the robot can, or asks for more than the robot can deliver. |
| **Joint velocity limit** | Maximum rotational speed a joint can safely reach. | §1 Mechanical #4 | Fast target changes (from the policy) that exceed this would be physically impossible to track. |
| **Link** | A rigid body segment between two joints (e.g. the upper arm, the shin). | §1 Mechanical #5 | Mass/geometry is defined per link, not per joint. |
| **Mass** | Weight of a single link, in kg. | §1 Mechanical #5 | Directly affects how much torque is needed to move/balance that link — wrong mass = wrong dynamics = policy doesn't transfer. |
| **Inertia tensor** | How a link's mass is distributed in space (resists rotation differently around different axes) — a 3×3 matrix, usually given as 3 diagonal values for a well-aligned body. | §1 Mechanical #5 | Determines how a link responds to torque (how fast it starts/stops rotating). Needed for realistic swing dynamics (arms, legs). |
| **Center of mass (CoM)** | The average position of a link's (or the whole robot's) mass — the point it balances around. | §1 Mechanical #5 | Balance/walking policies are extremely sensitive to CoM location; a torso CoM off by a few cm can change whether a policy can balance at all. |
| **Foot sole geometry / ground-contact layout** | The physical shape/points of the foot that actually touch the ground. | §1 Mechanical #7 | Defines the "support polygon" the robot balances over — this is one of the most balance-critical parameters in the whole spec. |
| **Ground-contact friction coefficient** | How much a foot resists sliding against the floor (unitless, roughly 0–2, higher = grippier). | §1 Mechanical #8 | Too little training-time friction range → policy is too cautious; too much → policy assumes grip the real floor won't give and slips. |

---

## 2. Actuator terms

| Term | What it means | Where it's needed | Why it matters |
|---|---|---|---|
| **Control mode** | How you command a joint: **position** (go to angle X), **velocity** (spin at rate X), **torque** (apply force X), or **impedance** (go to angle X, but "softly," like a spring). | §1 Actuators #9 | Our policy outputs *position* targets — this only works if the real joint controller accepts position (or impedance) commands, not torque-only. Determines whether the trained action-space design is even usable on this hardware. |
| **Native control-loop rate** | How many times per second the joint's own motor controller updates its output. | §1 Actuators #10 | Our simulated PD loop runs sub-steps at ~200 Hz between each 50 Hz policy command — the real controller needs to be fast enough to behave similarly, or the policy's implicit assumptions about "how quickly a target is reached" won't hold. |
| **Kp / Kd (PD gains)** | **Kp** ("stiffness") = how hard the joint pushes toward its target angle. **Kd** ("damping") = how much it resists velocity/oscillation. Together they form a PD (proportional-derivative) controller. | §1 Actuators #11 | The simulation bakes in specific Kp/Kd values per joint group when computing how far an action moves the joint (see §5 "scale"). Training with the wrong gains produces a policy tuned to dynamics the real joint doesn't have. |
| **Impedance-gain range** | The achievable range of "softness/stiffness" a joint controller can be set to, if it supports impedance control. | §1 Actuators #11 | Tells us whether our target Kp/Kd values are even physically achievable on the hardware, before we spend training time on them. |
| **Gear ratio** | How much the motor's raw rotation is reduced/amplified by a gearbox before it reaches the joint (e.g. 100:1 means the motor spins 100× for 1 joint rotation). | §1 Actuators #12 | Affects effective torque and speed at the joint, and how much backlash/friction shows up at the output — feeds the simulated actuator model. |
| **Transmission type** | The mechanism moving motor power to the joint — direct drive, belt, gearbox, parallel linkage (crank), etc. | §1 Actuators #12 | Some of Skandha's joints (ankles, some wrists) are driven through non-obvious linkages, not a direct 1:1 motor-to-joint mapping — this affects what "position target" even means mechanically, and therefore how the joint should be modeled in sim. |
| **Backlash** | Small "dead zone" of free play in a gear/linkage before it engages — the joint can move slightly without the motor moving. | §1 Actuators #12 | Adds a small unmodeled lag/inaccuracy between commanded and actual position that training-time noise should be sized to cover. |
| **Rotor / armature inertia** | The resistance-to-spin-up of the motor's own rotating parts (separate from the link it's attached to). | §1 Actuators #13 | Affects how quickly a joint can accelerate; the simulation currently uses one flat guess (`armature=0.01`) for every joint. |

---

## 3. Sensing terms

| Term | What it means | Where it's needed | Why it matters |
|---|---|---|---|
| **IMU (Inertial Measurement Unit)** | A sensor chip that measures rotation rate (gyroscope) and acceleration (accelerometer), sometimes fused into an orientation estimate. | §1 Sensing #14–15, §4 Model Input #1–2 | The policy's two most important balance signals (angular velocity, projected gravity) come from this sensor. |
| **Gyroscope (gyro)** | The part of the IMU measuring rotation rate (how fast the body is spinning), in rad/s per axis. | §4 Model Input #1 | Directly supplies "base angular velocity," the first 3 numbers the policy reads every tick. |
| **Accelerometer** | The part of the IMU measuring linear acceleration (including gravity) per axis. | §1 Sensing #15 | Used (with the gyro) to estimate orientation, which produces "projected gravity" (see §4). |
| **Noise density** | How "noisy"/jittery a sensor's raw readings are — a standard datasheet spec (e.g. rad/s/√Hz for a gyro). | §1 Sensing #15 | Training injects artificial sensor noise so the policy tolerates real-world imperfection — this value tells us whether our training-time noise range is realistic. |
| **Orientation estimate (attitude estimate)** | The robot's best guess of its own tilt/rotation in the world, usually as a quaternion or roll/pitch/yaw, computed by fusing gyro+accelerometer data over time. | §1 Sensing #15, §4 Model Input #2 | Needed to compute "projected gravity" — if the IMU doesn't output this itself, the deploy software must compute it, which changes what the training pipeline should assume as the observation source. |
| **Encoder** | A sensor on each joint that reports its current angle (and sometimes velocity). | §1 Sensing #16, §4 Model Input #5–6 | Supplies "joint positions" and "joint velocities" — 60 of the policy's 101 input numbers. |
| **Encoder resolution** | The smallest angle change an encoder can detect (e.g. 0.001 rad). | §1 Sensing #16 | Coarse resolution shows up as quantization noise in the joint-position observation — training-time position noise should be sized to at least cover it. |

---

## 4. Model Input terms (the observation vector)

| Term | What it means | Where it's needed | Why it matters |
|---|---|---|---|
| **Observation vector** | The full list of numbers the policy reads each tick to decide what to do (101 numbers for Skandha-Flat). | §2 Model Input | This is the policy's entire "view" of the robot and the world — nothing not in this list is seen by the policy. |
| **Base angular velocity** | How fast the torso is currently rotating, per axis (roll/pitch/yaw rate), in the torso's own frame. | §2 Model Input #1 | Core balance signal — tells the policy it's tipping before it's fallen. |
| **Body frame** | A coordinate system attached to and rotating with the robot's torso, rather than fixed to the world. | §2 Model Input #1–2 | Keeps the numbers meaningful regardless of which way the robot is currently facing. |
| **Projected gravity** | The direction "down" currently points, expressed in the torso's own frame — points straight down `(0,0,-1)`-ish when upright, tilts as the robot leans. | §2 Model Input #2 | This is effectively how the policy "feels" its own tilt/lean — the single most important balance observation. |
| **Velocity command** | The desired walking speed (forward/sideways) and turn rate, set externally by an operator or higher-level planner — not sensed from the robot at all. | §2 Model Input #3 | Tells the policy *what to do* (walk forward at 0.5 m/s, turn left, stand still, etc.). |
| **Gait phase (sin/cos clock)** | A repeating 0.6-second software timer, expressed as `sin`/`cos` so it wraps around smoothly, that resets to zero when the robot is commanded to stand still. | §2 Model Input #4 | Gives the policy a sense of walking rhythm (which foot should be swinging right now) without needing to sense it — purely a piece of software state that must be reproduced exactly during training and deployment alike. |
| **Joint position (relative to home pose)** | Each joint's current angle minus its trained "home"/default angle (see §5 below). | §2 Model Input #5 | Centers the input around the pose the policy was trained around, rather than raw hardware zero. |
| **Joint velocity** | Each joint's current rotation speed. | §2 Model Input #6 | Lets the policy react to fast motions, not just static position. |
| **Last action** | The exact 30 numbers the policy itself sent out on the *previous* tick, fed back in as part of this tick's input. | §2 Model Input #7 | Gives the policy short-term memory of its own recent commands, which helps produce smoother motion — this is pure software bookkeeping, not a sensor reading. |

---

## 5. Model Output terms (the action vector)

| Term | What it means | Where it's needed | Why it matters |
|---|---|---|---|
| **Action vector** | The 30 numbers the policy outputs each tick, one per joint. | §3 Model Output | This is everything the policy is allowed to control — nothing else. |
| **Raw action (`a`)** | The policy's direct output value for one joint, before any conversion — roughly in a small range like -1 to 1. | §3 Model Output | Not a usable angle by itself — must be converted using the formula below. |
| **Home angle / default pose** | The fixed reference joint angle (a slight crouch stance) that all position observations and action targets are measured relative to. | §3 Model Output | Both inputs and outputs are "relative," so this pose is defined once during training and must stay consistent for the mapping to mean anything. |
| **Scale** | A fixed per-joint multiplier converting a raw action into a real angle change (`target = home + a × scale`). Derived from that joint's torque limit and Kp during training. | §3 Model Output | Sets how big a movement one unit of policy output actually produces — this is computed directly from the mechanical/actuator parameters above (torque limit, Kp), so getting those right upstream is what makes this correct. |
| **Target angle** | The final joint angle the policy wants that joint to reach right now, computed from home angle + raw action × scale. | §3 Model Output | This is the number the training-time PD controller (and later, the real joint controller) tracks. |
| **PD tracking** | The controller's job of driving the joint toward the target angle using its own Kp/Kd loop, rather than being told a torque directly. | §3 Model Output | The policy assumes *something* smooths out the step from current angle to target angle — in training, that's the simulated PD loop built from the Kp/Kd values above. |

---

## 6. Timing terms

| Term | What it means | Where it's needed | Why it matters |
|---|---|---|---|
| **Policy inference rate** | How often the policy network runs and produces a new 30-value action (50 Hz = every 20 ms). | §4 Timing | Sets the pace of the whole control loop — both the input vector and output vector are refreshed at this rate during training. |
| **Decimation** | How many low-level physics/control sub-steps happen between each policy tick (4, in simulation) — i.e. the policy "holds" its target for several fast control cycles. | §4 Timing | Explains why the joint-level control loop is simulated at a faster rate (~200 Hz) than the policy itself (50 Hz) — this ratio is baked into training and should be matched at deployment. |
