"""Script to play RL agent with RSL-RL."""

import os
import sys

# Headless Linux/Kaggle support: select EGL before anything imports MuJoCo.
# MuJoCo captures MUJOCO_GL during import, so setting it later is too late.
# This only changes the backend on Linux machines that have no display;
# existing native/Viser behavior on desktop systems is preserved.
if sys.platform.startswith("linux") and not (
  os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
):
  os.environ.setdefault("MUJOCO_GL", "egl")
  print(f"[INFO] Headless Linux detected: MUJOCO_GL={os.environ['MUJOCO_GL']}")

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser", "kaggle"] = "auto"
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""

  # ---------------------------------------------------------------------------
  # Kaggle/headless playback options.
  # These are only used when --viewer kaggle is selected.
  # Existing native/viser/video behavior is unchanged.
  # ---------------------------------------------------------------------------
  kaggle_output: str = "kaggle_play.mp4"
  kaggle_fps: int | None = None
  kaggle_display: bool = True

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


def _run_kaggle_viewer(
  env: RslRlVecEnvWrapper,
  policy,
  video_length: int,
  output_file: str,
  fps: int | None,
  display_video: bool,
) -> None:
  """Run a headless rollout and display the RGB render inside Kaggle/Jupyter.

  This backend never creates a GLFW/native window. It uses mjlab's
  rgb_array/offscreen renderer, writes an MP4, and optionally embeds it
  directly in the notebook output.
  """
  import imageio.v2 as imageio
  from IPython.display import HTML, display

  # The RSL-RL wrapper resets the environment during construction, so
  # observations are already available here.
  obs = env.get_observations()

  if fps is None:
    step_dt = getattr(env.unwrapped, "step_dt", None)
    if step_dt is not None and step_dt > 0:
      fps = max(1, round(1.0 / float(step_dt)))
    else:
      fps = 30

  output_path = Path(output_file)
  output_path.parent.mkdir(parents=True, exist_ok=True)

  frames = []
  print(
    f"[INFO] Kaggle headless viewer: rendering {video_length} steps "
    f"at {fps} FPS"
  )

  for step in range(video_length):
    with torch.inference_mode():
      actions = policy(obs)

    obs, _, dones, _ = env.step(actions)

    # ManagerBasedRlEnv.render() supports rgb_array without GLFW.
    frame = env.unwrapped.render()
    if frame is not None:
      frames.append(frame)

    if torch.any(dones).item():
      # The environment normally auto-resets terminated instances.
      # We intentionally continue so the requested video length is reached.
      pass

  if not frames:
    raise RuntimeError(
      "Kaggle viewer produced no RGB frames. "
      "Make sure --viewer kaggle is used so render_mode='rgb_array' is enabled."
    )

  imageio.mimwrite(
    output_path,
    frames,
    fps=fps,
    codec="libx264",
    quality=8,
  )

  print(f"[INFO] Kaggle video saved to: {output_path.resolve()}")

  if display_video:
    import base64

    video_bytes = output_path.read_bytes()
    video_b64 = base64.b64encode(video_bytes).decode("ascii")
    display(
      HTML(
        f"""
        <div style="margin-top:12px">
          <h3>G1 Simulation</h3>
          <video controls autoplay loop style="max-width:100%; height:auto;">
            <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
            Your browser does not support embedded MP4 video.
          </video>
        </div>
        """
      )
    )


class _LocomotionController:
  """Keyboard locomotion controller for the native MuJoCo viewer.

  Maps WASD / QE / arrow keys to velocity commands that are written directly
  into the environment's ``twist`` command tensor each time a key is pressed
  or released.  The controller is passed as the ``key_callback`` argument of
  :class:`~mjlab.viewer.NativeMujocoViewer`.

  Key bindings
  ------------
  W / Up    : +forward  (lin_vel_x)
  S / Down  : -forward  (lin_vel_x)
  A / Left  : +strafe left  (lin_vel_y)
  D / Right : -strafe right (lin_vel_y)
  Q         : +yaw (turn left)
  E         : -yaw (turn right)
  X         : stop (zero all commands)

  Each axis is set to a fixed magnitude while the key is held and cleared when
  released.  Multiple keys can be held simultaneously.
  """

  _DEFAULT_LIN = 1.0  # m/s fallback
  _DEFAULT_ANG = 1.0  # rad/s fallback

  def __init__(self, env: "RslRlVecEnvWrapper") -> None:
    self._env = env
    self._held: set[int] = set()

    # Resolve speed magnitudes from the env command ranges when available.
    try:
      twist_cfg = env.unwrapped.cfg.commands["twist"]
      r = twist_cfg.ranges
      self._lin_x_mag = float(max(abs(r.lin_vel_x[0]), abs(r.lin_vel_x[1])))
      self._lin_y_mag = float(max(abs(r.lin_vel_y[0]), abs(r.lin_vel_y[1])))
      self._ang_z_mag = float(max(abs(r.ang_vel_z[0]), abs(r.ang_vel_z[1])))
    except Exception:
      self._lin_x_mag = self._DEFAULT_LIN
      self._lin_y_mag = self._DEFAULT_LIN
      self._ang_z_mag = self._DEFAULT_ANG

    print(
      "\n[LocoCtrl] Locomotion keys active:\n"
      "  W/Up=forward   S/Down=back\n"
      "  A/Left=strafe-left   D/Right=strafe-right\n"
      "  Q=turn-left    E=turn-right    X=stop\n"
    )

  def on_key(self, key: int) -> None:
    """Toggle held state for movement keys; write command each press/release."""
    from mjlab.viewer.native.keys import (
      KEY_A, KEY_D, KEY_E, KEY_Q, KEY_S, KEY_W, KEY_X,
      KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT,
    )

    if key == KEY_X:
      self._held.clear()
      self._apply()
      return

    movement_keys = {KEY_W, KEY_S, KEY_A, KEY_D, KEY_Q, KEY_E,
                     KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT}
    if key in movement_keys:
      if key in self._held:
        self._held.discard(key)
      else:
        self._held.add(key)
      self._apply()

  def _apply(self) -> None:
    """Resolve held keys into (vx, vy, wz) and write to the twist command."""
    from mjlab.viewer.native.keys import (
      KEY_A, KEY_D, KEY_E, KEY_Q, KEY_S, KEY_W,
      KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT,
    )
    h = self._held
    vx = (
      (self._lin_x_mag if (KEY_W in h or KEY_UP in h) else 0.0)
      - (self._lin_x_mag if (KEY_S in h or KEY_DOWN in h) else 0.0)
    )
    vy = (
      (self._lin_y_mag if (KEY_A in h or KEY_LEFT in h) else 0.0)
      - (self._lin_y_mag if (KEY_D in h or KEY_RIGHT in h) else 0.0)
    )
    wz = (
      (self._ang_z_mag if KEY_Q in h else 0.0)
      - (self._ang_z_mag if KEY_E in h else 0.0)
    )
    try:
      twist_term = self._env.unwrapped.command_manager.get_term("twist")
      twist_term.vel_command_b[:, 0] = vx
      twist_term.vel_command_b[:, 1] = vy
      twist_term.vel_command_b[:, 2] = wz
      # Freeze the resample timer so the command is not overwritten.
      twist_term.time_left[:] = 1e6
      # Prevent the standing-env zero-clamp from overriding our command.
      twist_term.is_standing_env[:] = False
    except Exception:
      pass  # Non-velocity tasks silently ignore locomotion control.


class _ViserWithPush(ViserPlayViewer):
  """ViserPlayViewer subclass that adds a push/perturbation panel.

  Mirrors the mouse-perturbation capability of the native MuJoCo viewer.

  Push modes
  ----------
  *Velocity impulse* (one-shot):
      Adds a random velocity kick sampled uniformly from [-mag, +mag] on the
      selected axes and writes it via ``write_root_link_velocity_to_sim``.
      Identical to the training-time ``push_by_setting_velocity`` event.

  *Sustained wrench* (hold):
      Applies a constant body-frame force [fx, fy, fz] to the robot's root
      body for ``duration`` seconds via ``write_external_wrench_to_sim``.
      A background timer zeros it out once the duration expires.

  Both modes operate on the currently-selected environment (``env_idx``).
  """

  def setup(self) -> None:  # noqa: D102
    super().setup()
    self._push_timer: float = 0.0       # Seconds remaining on sustained push.
    self._push_active: bool = False
    self._push_env_idx: int = 0         # Snapshot of env_idx at push time.
    self._setup_push_gui()

  def _setup_push_gui(self) -> None:
    import viser
    server = self._server

    with server.gui.add_folder("Push Robot"):
      # ── mode selector ──────────────────────────────────────────────
      self._push_mode = server.gui.add_dropdown(
        "Mode",
        options=["Velocity Impulse", "Sustained Wrench"],
        initial_value="Velocity Impulse",
      )

      # ── velocity-impulse controls ───────────────────────────────────
      with server.gui.add_folder("Velocity Impulse"):
        self._push_vx = server.gui.add_slider(
          "Vx (m/s)", min=-3.0, max=3.0, step=0.1, initial_value=1.5
        )
        self._push_vy = server.gui.add_slider(
          "Vy (m/s)", min=-3.0, max=3.0, step=0.1, initial_value=0.0
        )
        self._push_vz = server.gui.add_slider(
          "Vz (m/s)", min=-2.0, max=2.0, step=0.1, initial_value=0.0
        )
        self._push_random = server.gui.add_checkbox(
          "Randomise direction", initial_value=False
        )
        push_btn = server.gui.add_button(
          "Push!", icon=viser.Icon.HAND_STOP
        )

        @push_btn.on_click
        def _(_) -> None:
          self._do_velocity_impulse()

      # ── sustained-wrench controls ───────────────────────────────────
      with server.gui.add_folder("Sustained Wrench"):
        self._wrench_fx = server.gui.add_slider(
          "Fx (N)", min=-500.0, max=500.0, step=5.0, initial_value=200.0
        )
        self._wrench_fy = server.gui.add_slider(
          "Fy (N)", min=-500.0, max=500.0, step=5.0, initial_value=0.0
        )
        self._wrench_fz = server.gui.add_slider(
          "Fz (N)", min=-500.0, max=500.0, step=5.0, initial_value=0.0
        )
        self._wrench_dur = server.gui.add_slider(
          "Duration (s)", min=0.1, max=5.0, step=0.1, initial_value=0.5
        )
        self._push_status = server.gui.add_text(
          "Status", initial_value="Idle"
        )
        apply_btn = server.gui.add_button(
          "Apply Wrench", icon=viser.Icon.ARROW_BIG_RIGHT_FILLED
        )
        cancel_btn = server.gui.add_button(
          "Cancel", icon=viser.Icon.PLAYER_STOP
        )

        @apply_btn.on_click
        def _(_) -> None:
          self._do_sustained_wrench()

        @cancel_btn.on_click
        def _(_) -> None:
          self._cancel_wrench()

  # ------------------------------------------------------------------
  # One-shot velocity impulse
  # ------------------------------------------------------------------

  def _do_velocity_impulse(self) -> None:
    try:
      import torch
      env = self.env.unwrapped
      robot = env.scene["robot"]
      env_idx = self._scene.env_idx
      env_ids = torch.tensor([env_idx], device=env.device, dtype=torch.long)

      vel_w = robot.data.root_link_vel_w[env_ids].clone()

      if self._push_random.value:
        # Uniform random in configured training ranges.
        try:
          vr = env.cfg.commands["twist"].ranges
          vx = float(torch.empty(1).uniform_(vr.lin_vel_x[0], vr.lin_vel_x[1]))
          vy = float(torch.empty(1).uniform_(vr.lin_vel_y[0], vr.lin_vel_y[1]))
          vz = 0.0
        except Exception:
          vx = float(torch.empty(1).uniform_(-1.5, 1.5))
          vy = float(torch.empty(1).uniform_(-1.5, 1.5))
          vz = 0.0
      else:
        vx = float(self._push_vx.value)
        vy = float(self._push_vy.value)
        vz = float(self._push_vz.value)

      vel_w[:, 0] += vx
      vel_w[:, 1] += vy
      vel_w[:, 2] += vz
      robot.write_root_link_velocity_to_sim(vel_w, env_ids=env_ids)
      print(f"[Push] Velocity impulse → vx={vx:+.2f} vy={vy:+.2f} vz={vz:+.2f}")
    except Exception as exc:
      print(f"[Push] Velocity impulse failed: {exc}")

  # ------------------------------------------------------------------
  # Sustained external wrench
  # ------------------------------------------------------------------

  def _do_sustained_wrench(self) -> None:
    try:
      import torch
      env = self.env.unwrapped
      robot = env.scene["robot"]
      env_idx = self._scene.env_idx
      env_ids = torch.tensor([env_idx], device=env.device, dtype=torch.long)

      fx = float(self._wrench_fx.value)
      fy = float(self._wrench_fy.value)
      fz = float(self._wrench_fz.value)
      dur = float(self._wrench_dur.value)

      forces = torch.tensor([[[fx, fy, fz]]], device=env.device)   # (1,1,3)
      torques = torch.zeros_like(forces)
      robot.write_external_wrench_to_sim(forces, torques, env_ids=env_ids)

      self._push_active = True
      self._push_timer = dur
      self._push_env_idx = env_idx
      self._push_status.value = f"Active ({dur:.1f}s)"
      print(f"[Push] Sustained wrench → fx={fx:+.0f} fy={fy:+.0f} fz={fz:+.0f} N for {dur:.1f}s")
    except Exception as exc:
      print(f"[Push] Sustained wrench failed: {exc}")

  def _cancel_wrench(self) -> None:
    self._push_active = False
    self._push_timer = 0.0
    self._push_status.value = "Idle"
    try:
      import torch
      env = self.env.unwrapped
      robot = env.scene["robot"]
      env_ids = torch.tensor([self._push_env_idx], device=env.device, dtype=torch.long)
      zeros = torch.zeros((1, 1, 3), device=env.device)
      robot.write_external_wrench_to_sim(zeros, zeros, env_ids=env_ids)
    except Exception:
      pass

  # ------------------------------------------------------------------
  # Tick the wrench timer on every physics step
  # ------------------------------------------------------------------

  def sync_viewer_to_env(self) -> None:  # noqa: D102
    if not self._push_active:
      return
    step_dt = self.env.unwrapped.step_dt
    self._push_timer -= step_dt
    if self._push_timer <= 0.0:
      self._cancel_wrench()
      print("[Push] Wrench expired.")


def run_play(task_id: str, cfg: PlayConfig):


  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check for local motion file first (works for both dummy and trained modes).
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )

  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path)
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  # Existing video behavior is unchanged. The new Kaggle viewer additionally
  # requests rgb_array rendering so that no GLFW/X11 display is required.
  render_mode = (
    "rgb_array"
    if ((TRAINED_MODE and cfg.video) or cfg.viewer == "kaggle")
    else None
  )

  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    loco_ctrl = _LocomotionController(env)
    NativeMujocoViewer(env, policy, key_callback=loco_ctrl.on_key).run()
  elif resolved_viewer == "viser":
    _ViserWithPush(env, policy).run()
  elif resolved_viewer == "kaggle":
    _run_kaggle_viewer(
      env=env,
      policy=policy,
      video_length=cfg.video_length,
      output_file=cfg.kaggle_output,
      fps=cfg.kaggle_fps,
      display_video=cfg.kaggle_display,
    )
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401
  import src.tasks

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
