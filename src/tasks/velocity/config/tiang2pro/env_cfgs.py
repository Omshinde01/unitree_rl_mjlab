"""Tiangong / TienKung Pro velocity environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from src.assets.robots import get_tienkung_pro_robot_cfg , TIENKUNG_PRO_ACTION_SCALE
from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def tiangong_rough_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Tiangong Pro rough-terrain velocity configuration."""

    cfg = make_velocity_env_cfg()

    # ========================================================================
    # Simulation
    # ========================================================================

    cfg.sim.mujoco.ccd_iterations = 500
    cfg.sim.contact_sensor_maxmatch = 500
    cfg.sim.nconmax = 48

    # ========================================================================
    # Robot
    # ========================================================================

    cfg.scene.entities = {
        "robot": get_tienkung_pro_robot_cfg(),
    }

    # ========================================================================
    # Terrain scan
    # ========================================================================

    for sensor in cfg.scene.sensors or ():
        if sensor.name == "terrain_scan":
            assert isinstance(sensor, RayCastSensorCfg)
            sensor.frame.name = "pelvis"

    # ========================================================================
    # Foot configuration
    # ========================================================================

    foot_body_names = (
        "ankle_roll_l_link",
        "ankle_roll_r_link",
    )

    foot_geom_names = (
        "ankle_roll_l_link_collision",
        "ankle_roll_r_link_collision",
    )

    # ========================================================================
    # Contact sensors
    # ========================================================================

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^ankle_roll_[lr]_link$",
            entity="robot",
        ),
        secondary=ContactMatch(
            mode="body",
            pattern="terrain",
        ),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(
            mode="subtree",
            pattern="pelvis",
            entity="robot",
        ),
        secondary=ContactMatch(
            mode="subtree",
            pattern="pelvis",
            entity="robot",
        ),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )

    cfg.scene.sensors = (
        cfg.scene.sensors or ()
    ) + (
        feet_ground_cfg,
        self_collision_cfg,
    )

    # ========================================================================
    # Terrain curriculum
    # ========================================================================

    if (
        cfg.scene.terrain is not None
        and cfg.scene.terrain.terrain_generator is not None
    ):
        cfg.scene.terrain.terrain_generator.curriculum = True

    # ========================================================================
    # Joint-position action
    # ========================================================================

    joint_pos_action = cfg.actions["joint_pos"]

    assert isinstance(
        joint_pos_action,
        JointPositionActionCfg,
    )

    # The generic velocity environment already uses:
    #
    #     entity_name="robot"
    #     actuator_names=(".*",)
    #     scale=0.25
    #     use_default_offset=True
    #
    # Therefore all actuators defined by the Tiangong articulation are
    # automatically selected here.
    #
    # Tiangong articulation:
    #
    #     Legs       = 12
    #     Waist      =  1
    #     Head       =  3
    #     Arms       = 12
    #     -----------------
    #     Total      = 28
    #
    # Do not add wrist_pitch joints: those joints were removed from the
    # Tiangong MJCF.

    joint_pos_action.actuator_names = (".*",)
    joint_pos_action.use_default_offset = True
    joint_pos_action.scale = TIENKUNG_PRO_ACTION_SCALE

    # ========================================================================
    # Viewer
    # ========================================================================

    cfg.viewer.body_name = "pelvis"

    # ========================================================================
    # Velocity command
    # ========================================================================

    twist_cmd = cfg.commands["twist"]

    assert isinstance(
        twist_cmd,
        UniformVelocityCommandCfg,
    )

    twist_cmd.viz.z_offset = 1.15

    # ========================================================================
    # Critic: foot height
    # ========================================================================

    cfg.observations["critic"].terms[
        "foot_height"
    ].params["asset_cfg"].site_names = None

    cfg.observations["critic"].terms[
        "foot_height"
    ].params["asset_cfg"].body_names = foot_body_names

    # ========================================================================
    # Randomization
    # ========================================================================

    cfg.events["foot_friction"].params[
        "asset_cfg"
    ].geom_names = foot_geom_names

    cfg.events["base_com"].params[
        "asset_cfg"
    ].body_names = ("pelvis",)

    # ========================================================================
    # Pose reward
    # ========================================================================

    cfg.rewards["pose"].params["std_standing"] = {
        r"hip_pitch_.*_joint": 0.05,
        r"hip_roll_.*_joint": 0.05,
        r"hip_yaw_.*_joint": 0.05,
        r"knee_pitch_.*_joint": 0.05,
        r"ankle_roll_.*_joint": 0.05,
        r"ankle_pitch_.*_joint": 0.05,
        r"body_yaw_joint": 0.05,
        r"head_yaw_joint": 0.05,
        r"head_pitch_joint": 0.05,
        r"head_roll_joint": 0.05,
        r"shoulder_pitch_.*_joint": 0.05,
        r"shoulder_roll_.*_joint": 0.05,
        r"shoulder_yaw_.*_joint": 0.05,
        r"elbow_pitch_.*_joint": 0.05,
        r"elbow_yaw_.*_joint": 0.05,
        r"wrist_roll_.*_joint": 0.05,
    }

    cfg.rewards["pose"].params["std_walking"] = {
        r"hip_pitch_.*_joint": 0.5,
        r"hip_roll_.*_joint": 0.15,
        r"hip_yaw_.*_joint": 0.15,
        r"knee_pitch_.*_joint": 0.5,
        r"ankle_roll_.*_joint": 0.1,
        r"ankle_pitch_.*_joint": 0.15,
        r"body_yaw_joint": 0.15,
        r"head_yaw_joint": 0.15,
        r"head_pitch_joint": 0.15,
        r"head_roll_joint": 0.15,
        r"shoulder_pitch_.*_joint": 0.15,
        r"shoulder_roll_.*_joint": 0.1,
        r"shoulder_yaw_.*_joint": 0.1,
        r"elbow_pitch_.*_joint": 0.1,
        r"elbow_yaw_.*_joint": 0.1,
        r"wrist_roll_.*_joint": 0.1,
    }

    cfg.rewards["pose"].params["std_running"] = {
        r"hip_pitch_.*_joint": 0.5,
        r"hip_roll_.*_joint": 0.25,
        r"hip_yaw_.*_joint": 0.25,
        r"knee_pitch_.*_joint": 0.5,
        r"ankle_roll_.*_joint": 0.1,
        r"ankle_pitch_.*_joint": 0.25,
        r"body_yaw_joint": 0.25,
        r"head_yaw_joint": 0.25,
        r"head_pitch_joint": 0.25,
        r"head_roll_joint": 0.25,
        r"shoulder_pitch_.*_joint": 0.25,
        r"shoulder_roll_.*_joint": 0.1,
        r"shoulder_yaw_.*_joint": 0.1,
        r"elbow_pitch_.*_joint": 0.1,
        r"elbow_yaw_.*_joint": 0.1,
        r"wrist_roll_.*_joint": 0.1,
    }

    # ========================================================================
    # Body rewards
    # ========================================================================

    cfg.rewards[
        "body_orientation_l2"
    ].params["asset_cfg"].body_names = (
        "pelvis",
    )

    cfg.rewards[
        "body_ang_vel"
    ].params["asset_cfg"].body_names = (
        "pelvis",
    )

    # ========================================================================
    # Foot rewards
    # ========================================================================

    cfg.rewards[
        "foot_clearance"
    ].params["asset_cfg"].site_names = None

    cfg.rewards[
        "foot_clearance"
    ].params["asset_cfg"].body_names = foot_body_names

    cfg.rewards[
        "foot_slip"
    ].params["asset_cfg"].site_names = None

    cfg.rewards[
        "foot_slip"
    ].params["asset_cfg"].body_names = foot_body_names

    # ========================================================================
    # Self-collision reward
    # ========================================================================

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={
            "sensor_name": self_collision_cfg.name,
            "force_threshold": 10.0,
        },
    )
    cfg.rewards["track_linear_velocity"].weight = 1.5
    # ========================================================================
    # Play mode
    # ========================================================================

    if play:
        cfg.episode_length_s = int(1e9)

        cfg.observations[
            "actor"
        ].enable_corruption = False

        cfg.events.pop(
            "push_robot",
            None,
        )

        cfg.curriculum = {}

        cfg.events["randomize_terrain"] = EventTermCfg(
            func=envs_mdp.randomize_terrain,
            mode="reset",
            params={},
        )

        if cfg.scene.terrain is not None:
            if (
                cfg.scene.terrain.terrain_generator
                is not None
            ):
                cfg.scene.terrain.terrain_generator.curriculum = False
                cfg.scene.terrain.terrain_generator.num_cols = 5
                cfg.scene.terrain.terrain_generator.num_rows = 5
                cfg.scene.terrain.terrain_generator.border_width = 10.0

    return cfg


def tiangong_flat_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Tiangong Pro flat-terrain velocity configuration."""

    cfg = tiangong_rough_env_cfg(
        play=play,
    )

    # ========================================================================
    # Flat simulation
    # ========================================================================

    cfg.sim.njmax = 300
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 64
    cfg.sim.nconmax = None

    # ========================================================================
    # Flat terrain
    # ========================================================================

    assert cfg.scene.terrain is not None

    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # ========================================================================
    # Remove terrain scan
    # ========================================================================

    cfg.scene.sensors = tuple(
        sensor
        for sensor in (cfg.scene.sensors or ())
        if sensor.name != "terrain_scan"
    )

    del cfg.observations[
        "actor"
    ].terms["height_scan"]

    del cfg.observations[
        "critic"
    ].terms["height_scan"]

    # ========================================================================
    # Disable terrain curriculum
    # ========================================================================

    cfg.curriculum.pop(
        "terrain_levels",
        None,
    )

    # ========================================================================
    # Flat-play command ranges
    # ========================================================================

    if play:
        twist_cmd = cfg.commands["twist"]

        assert isinstance(
            twist_cmd,
            UniformVelocityCommandCfg,
        )

        twist_cmd.ranges.lin_vel_x = (
            -0.5,
            1.0,
        )

        twist_cmd.ranges.lin_vel_y = (
            -0.5,
            0.5,
        )

        twist_cmd.ranges.ang_vel_z = (
            -0.5,
            0.5,
        )

    return cfg