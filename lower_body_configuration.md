# SkandhaLowerBody — Changes vs. Full Skandha Robot

## Removed
- Both 7-DOF arms (`RA_*`, `LA_*` bodies/joints)
- 3-DOF head (`head_link_1/2/3`, `head_camera`)
- 3 chest-camera dummy bodies (`lower_chest_camera`, `upper_chest_front_camera`, `upper_chest_rear_camera`)
- All joint sensors (`jointpos`/`jointvel`/`jointactuatorfrc`) for the removed joints
- Unused mesh assets for removed bodies
- 6 self-collision `<exclude>` pairs that referenced removed bodies

## Kept unchanged
- Torso (`base_link`) — same mass, inertia, mesh (still the floating base)
- `hip_link` + `waist_joint`
- Both full 6-DOF legs
- Hip-mounted IMU bracket bodies (incl. `vectornav_hip_front_site_imu`)
- All surviving joints' ranges, torque limits, PD gains — copied verbatim, not re-tuned
- Home-keyframe base height + leg/waist joint angles — reused as-is, re-verified flat-footed

## Key numbers

| | Full Skandha | SkandhaLowerBody |
|---|---|---|
| DOF | 30 | 13 (legs ×12 + waist ×1) |
| Total mass | ~56.2 kg | ~45.1 kg |
| Actor obs (flat) | 101 | 50 |
| Critic obs (flat) | 116 | 65 |
| Action dim | 30 | 13 |

## Not changed
- No joint limit, torque limit, or gain value was altered on any surviving joint
- No new reward/observation term added — only arm/head entries dropped from the `pose` reward's std dict

## New files (full robot untouched)
- `src/assets/robots/skandhalowerbody/` (XML, assets, constants, README)
- `src/tasks/velocity/config/skandhalowerbody/` (env cfg, RL cfg, task registration)
- 1 appended import block in `src/assets/robots/__init__.py`

## Open flag
`base_link` mass/inertia was **not** increased to compensate for removed arm/head
mass — needs a robotics-team decision if the real rig keeps them on as dead weight
instead of physically removing them.
