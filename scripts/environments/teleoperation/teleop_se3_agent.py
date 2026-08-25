# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run teleoperation with Isaac Lab manipulation environments.

Supports multiple input devices (e.g., keyboard, spacemouse, gamepad) and devices
configured within the environment (including OpenXR-based hand tracking or motion
controllers), all driven through the IsaacTeleop session/pipeline API
(:mod:`isaaclab_teleop`). Pass ``--teleop_device`` to select a built-in device by name,
or declare ``env_cfg.isaac_teleop`` in the environment config for a custom pipeline.
"""

"""Launch Isaac Sim Simulator first."""

# Isaac Lab does not use Warp autodiff; skipping adjoint codegen roughly halves the
# time spent building kernels on a cold kernel cache.
import warp as wp

wp.config.enable_backward = False

import argparse
from collections.abc import Callable

from isaaclab.app import AppLauncher
from isaaclab.utils.string import list_intersection, string_to_callable

# add argparse arguments
parser = argparse.ArgumentParser(description="Teleoperation for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default=None,
    help=(
        "Built-in device name (keyboard, spacemouse, gamepad) to drive teleoperation with,"
        " overriding any isaac_teleop pipeline declared in the environment config. When omitted,"
        " the environment's declared isaac_teleop pipeline is used if present, otherwise keyboard"
        " is used as a fallback."
    ),
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
parser.add_argument(
    "--cloudxr_env",
    type=str,
    default=None,
    help=(
        "Path to a CloudXR .env file, or a shorthand: 'cloudxrjs' (Quest/Pico), 'avp' (Apple Vision Pro),"
        " or 'standalone' (headless, no XR client). Set to 'none' to disable CloudXR auto-launch entirely."
        " When unset, defaults to 'cloudxrjs' with --xr and 'standalone' without --xr."
    ),
)
parser.add_argument(
    "--auto_launch_cloudxr",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Auto-launch the CloudXR runtime when --cloudxr_env is set. Use --no-auto_launch_cloudxr to disable.",
)
parser.add_argument(
    "--enable_debug_visualization",
    action="store_true",
    default=False,
    help="Enable hand joint and controller aim debug visualization at session start (IsaacTeleop only).",
)
parser.add_argument(
    "--external_callback",
    default=None,
    help="Fully qualified path to an externally defined callback.",
)
parser.add_argument(
    "--disable_external_cameras",
    action="store_true",
    default=False,
    help=(
        "Disable external camera rendering. External cameras render by default for teleoperation;"
        " pass this flag to strip camera sensors from the environment (e.g. to reduce GPU contention"
        " and improve XR performance)."
    ),
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, remaining_args = parser.parse_known_args()

app_launcher_args = vars(args_cli)

# Enable external camera rendering by default (``--disable_external_cameras`` turns it off). The
# ``--enable_cameras`` CLI flag was removed in Isaac Lab 3.0 (see #6656), so pass the intent to
# AppLauncher as a kwarg; this selects a camera-rendering experience that provides RTX/DLSS support.
# Everywhere else we read ``args_cli.disable_external_cameras`` directly.
app_launcher = AppLauncher(app_launcher_args, enable_cameras=not args_cli.disable_external_cameras)
simulation_app = app_launcher.app

# Call an external callback if requested.
remaining_args_env_registration = None
if args_cli.external_callback:
    external_callback_function = string_to_callable(args_cli.external_callback, separator=".")
    remaining_args_env_registration = external_callback_function()

# Error on unrecognized arguments.
unrecognized_args = list_intersection(remaining_args, remaining_args_env_registration)
if unrecognized_args:
    parser.error(f"unrecognized arguments: {' '.join(unrecognized_args)}")

"""Rest everything follows."""


import logging

import gymnasium as gym
import torch
from isaaclab_physx.renderers import IsaacRtxRendererGlobalSettingsCfg
from isaaclab_physx.renderers.isaac_rtx_renderer_utils import (
    apply_isaac_rtx_global_settings,
)
from isaaclab_teleop.builtin_teleop import SE3_TELEOP_CFG_BUILDERS
from isaaclab_teleop.control_pollers import KeyboardControlPoller, SpaceMouseResetPoller

from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.core.lift import mdp
from isaaclab_tasks.utils import parse_env_cfg

logger = logging.getLogger(__name__)

_CLOUDXR_ENV_SHORTHANDS: dict[str, str] = {}


def _resolve_cloudxr_env(value: str | None, xr_enabled: bool = False) -> str | None:
    """Resolve ``--cloudxr_env`` shorthands to absolute ``.env`` file paths.

    Accepts ``"cloudxrjs"`` (Quest/Pico), ``"avp"`` (Apple Vision Pro),
    ``"standalone"`` (headless, no XR client), ``"none"`` (disable), or an
    arbitrary file path. When *value* is ``None`` (flag unset), defaults to
    ``"cloudxrjs"`` when *xr_enabled* else ``"standalone"`` -- so a run without
    ``--xr`` uses the clientless headless profile.
    """
    if value is None:
        value = "cloudxrjs" if xr_enabled else "standalone"
    if value.strip() == "" or value.lower() == "none":
        return None
    if not _CLOUDXR_ENV_SHORTHANDS:
        from isaaclab_teleop import CLOUDXR_AVP_ENV, CLOUDXR_JS_ENV, CLOUDXR_STANDALONE_ENV

        _CLOUDXR_ENV_SHORTHANDS["cloudxrjs"] = CLOUDXR_JS_ENV
        _CLOUDXR_ENV_SHORTHANDS["avp"] = CLOUDXR_AVP_ENV
        _CLOUDXR_ENV_SHORTHANDS["standalone"] = CLOUDXR_STANDALONE_ENV
    return _CLOUDXR_ENV_SHORTHANDS.get(value.lower(), value)


def _rtx_rendering_requested(args: argparse.Namespace) -> bool:
    """Return whether the CLI selects a renderer that actually drives RTX rendering.

    The RTX/DLSS global settings are only meaningful when something renders through RTX.
    That happens when the Kit visualizer is enabled (``--viz kit``), when external cameras
    are rendered (on by default; see ``--disable_external_cameras``), or in XR mode (``--xr``).
    A pure-headless session with none of these renders nothing.

    This reads the resolved namespace intent rather than any Kit/carb runtime state so the
    check keeps working as these scripts grow support for other renderers and kitless runs.
    """
    visualizers = getattr(args, "visualizer", None) or []
    external_cameras = not getattr(args, "disable_external_cameras", False)
    return external_cameras or ("kit" in visualizers) or bool(getattr(args, "xr", False))


def _ensure_replicator_loaded() -> None:
    """Enable ``omni.replicator.core`` so RTX/DLSS global settings can be applied.

    :func:`apply_isaac_rtx_global_settings` sets the antialiasing mode through
    ``omni.replicator.core``, which ships with the SDG/rendering extensions. Some Kit
    experiences (e.g. the Kit-viewport-only app selected by ``--visualizer kit`` without
    cameras or XR) do not preload it, so enable it on demand via the extension manager
    before applying RTX settings. Idempotent when the extension is already enabled.
    """
    import omni.kit.app

    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.replicator.core", True)


#: Sensitivity multiplier applied to a built-in device's default pos/rot sensitivity when
#: selected via --teleop_device, matching historical per-device tuning.
_BUILTIN_SENSITIVITY_SCALE = {"keyboard": 0.05, "spacemouse": 0.05, "gamepad": 0.1}


def _resolve_isaac_teleop_cfg(env_cfg, device_name: str | None, sensitivity: float, sim_device: str):
    """Resolve the IsaacTeleopCfg to drive teleoperation with.

    ``--teleop_device`` (if given) always wins; otherwise the environment's declared
    ``isaac_teleop`` pipeline is used; otherwise keyboard is a fallback default. Returns
    ``None`` if ``device_name`` is set but not a recognized built-in device.
    """
    if device_name is None and getattr(env_cfg, "isaac_teleop", None) is not None:
        return env_cfg.isaac_teleop
    name = (device_name or "keyboard").lower()
    builder = SE3_TELEOP_CFG_BUILDERS.get(name)
    if builder is None:
        return None
    scale = _BUILTIN_SENSITIVITY_SCALE.get(name, 1.0)
    return builder(pos_sensitivity=scale * sensitivity, rot_sensitivity=scale * sensitivity, sim_device=sim_device)


def _make_haptic_io(env, teleop_interface, env_cfg):
    """Return ``(update, stop)`` callables driving controller haptics, or no-ops.

    Keeps haptics opt-in without branching in the main loop: both callables are no-ops unless
    the env declares a ``haptic_feedback`` config. ``update`` renders the current contact
    force; ``stop`` zeroes it so a stale pulse does not persist while teleop is paused.
    """
    noop = lambda: None  # noqa: E731
    from isaaclab_teleop import create_haptic_feedback_driver

    driver = create_haptic_feedback_driver(env.unwrapped, teleop_interface, env_cfg)
    if driver is None:
        return noop, noop
    return driver.update, driver.stop


def _local_plugin_name(isaac_teleop_cfg) -> str | None:
    """The single bundled local-device plugin name driving this pipeline, or None for pure XR."""
    plugins = getattr(isaac_teleop_cfg, "plugins", None) or []
    names = {p.plugin_name for p in plugins}
    if not names:
        return None
    return next(iter(names))


class _AuxiliaryKeyboardPoller:
    """Advances an auxiliary keyboard-only IsaacTeleopDevice and its control poller together.

    Used for headset-free B/P/R control layered on top of a pure-XR primary pipeline (one with
    no bundled local-device plugin of its own).
    """

    def __init__(self, device, poller: KeyboardControlPoller) -> None:
        self._device = device
        self._poller = poller

    def advance(self) -> None:
        self._device.advance()
        self._poller.advance()


def _make_control_pollers(teleop_interface, isaac_teleop_cfg, has_window: bool, sim_device: str) -> list:
    """Build the pollers needed for headset-free physical control (B/P/R, or right-button reset).

    The caller must call ``.advance()`` on every returned poller every frame (in addition to
    advancing ``teleop_interface`` itself) and keep the list referenced for the app's lifetime.

    - A keyboard-plugin-backed primary pipeline is polled directly for B/P/R.
    - A spacemouse-plugin-backed primary pipeline is polled directly for its right-button reset.
    - A gamepad-plugin-backed primary pipeline has no physical control mechanism (matching the
      legacy gamepad device, which never wired one either) -- returns no pollers.
    - A pure-XR primary pipeline (no bundled plugin) gets an auxiliary keyboard-only device for
      B/P/R, since it has no physical input of its own to poll.

    Returns an empty list when there is no window (a windowless run still auto-starts teleop).
    """
    if not has_window:
        return []
    plugin_name = _local_plugin_name(isaac_teleop_cfg)
    if plugin_name == "keyboard":
        print("IsaacTeleop control keys: [B] start/resume  [P] pause  [R] reset")
        return [KeyboardControlPoller(teleop_interface)]
    if plugin_name == "spacemouse":
        return [SpaceMouseResetPoller(teleop_interface)]
    if plugin_name is not None:
        # gamepad, or an unrecognized local plugin: no known physical control mechanism, and
        # layering a second plugin-backed session here would collide with the primary's
        # CloudXR runtime the same way a second keyboard session would.
        return []

    try:
        from isaaclab_teleop import create_isaac_teleop_device
        from isaaclab_teleop.isaac_teleop_cfg import CLOUDXR_STANDALONE_ENV
        from isaaclab_teleop.keyboard import se3_keyboard_teleop_cfg

        aux_device = create_isaac_teleop_device(
            se3_keyboard_teleop_cfg(pos_sensitivity=0.0, rot_sensitivity=0.0, sim_device=sim_device),
            cloudxr_env_file=CLOUDXR_STANDALONE_ENV,
            use_kit_xr_bridge=False,
        )
        aux_device.__enter__()
        poller = KeyboardControlPoller(aux_device)
        poller.add_callback("B", teleop_interface.request_start)
        poller.add_callback("P", teleop_interface.request_stop)
        poller.add_callback("R", lambda: teleop_interface.reset(pause=True))
        print("IsaacTeleop control keys: [B] start/resume  [P] pause  [R] reset")
        return [_AuxiliaryKeyboardPoller(aux_device, poller)]
    except Exception as e:
        logger.warning(f"Control keyboard unavailable ({e}); teleop still auto-starts without --xr")
        return []


def main() -> None:  # noqa: C901
    """
    Run teleoperation with an Isaac Lab manipulation environment.

    Creates the environment, sets up teleoperation interfaces and callbacks,
    and runs the main simulation loop until the application is closed.

    Returns:
        None
    """
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise ValueError(
            "Teleoperation is only supported for ManagerBasedRLEnv environments. "
            f"Received environment config type: {type(env_cfg).__name__}"
        )
    # modify configuration
    env_cfg.terminations.time_out = None
    if "Lift" in args_cli.task:
        # set the resampling time range to large number to avoid resampling
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        # add termination condition for reaching the goal otherwise the environment won't reset
        env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)

    # --teleop_device (if given) always wins over any env-declared isaac_teleop pipeline;
    # otherwise the env's declared pipeline is used, falling back to keyboard if neither is set.
    isaac_teleop_cfg = _resolve_isaac_teleop_cfg(env_cfg, args_cli.teleop_device, args_cli.sensitivity, args_cli.device)
    if isaac_teleop_cfg is None:
        logger.error(
            f"--teleop_device={args_cli.teleop_device} is not a built-in device name."
            f" Built-in devices: {', '.join(sorted(SE3_TELEOP_CFG_BUILDERS))}."
        )
        simulation_app.close()
        return
    env_cfg.isaac_teleop = isaac_teleop_cfg

    from isaaclab_teleop import XrCameraFeedSession

    camera_feed_session = XrCameraFeedSession.prepare(
        env_cfg,
        enabled=args_cli.xr,
        camera_rendering_enabled=not args_cli.disable_external_cameras,
    )

    # XR-rendering setup (camera removal + DLSS) is only needed for the Kit XR
    # path. Without --xr, IsaacTeleop runs standalone (I/O only) and renders
    # normally, so gate on --xr alone.
    if args_cli.xr:
        # Keep camera configs when external cameras are enabled (defaulted on); otherwise
        # strip them so the XR headset view is the sole render product.
        if args_cli.disable_external_cameras:
            env_cfg = remove_camera_configs(env_cfg)
    # Apply the RTX/DLSS global settings when an RTX render pipeline will run (Kit visualizer,
    # external cameras, or XR). ``apply_isaac_rtx_global_settings`` uses ``omni.replicator``,
    # which some experiences do not preload, so ensure it is loaded first.
    if _rtx_rendering_requested(args_cli):
        _ensure_replicator_loaded()
        apply_isaac_rtx_global_settings(
            IsaacRtxRendererGlobalSettingsCfg(
                antialiasing_mode="DLSS",
                carb_settings=(
                    {"/rtx/dldenoiser/responsiveDenoising": True}
                    if camera_feed_session.requires_responsive_denoising
                    else None
                ),
            ),
        )

    try:
        # create environment
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        # check environment name (for reach , we don't allow the gripper)
        if "Reach" in args_cli.task:
            logger.warning(
                f"The environment '{args_cli.task}' does not support gripper control. The device command will be"
                " ignored."
            )
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        simulation_app.close()
        return

    # Flags for controlling teleoperation flow
    should_reset_recording_instance = False
    teleoperation_active = True

    # Callback handlers
    def reset_recording_instance() -> None:
        """
        Reset the environment to its initial state.

        Sets a flag to reset the environment on the next simulation step.

        Returns:
            None
        """
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        print("Reset triggered - Environment will reset on next step")

    def start_teleoperation() -> None:
        """
        Activate teleoperation control of the robot.

        Enables the application of teleoperation commands to the environment.

        Returns:
            None
        """
        nonlocal teleoperation_active
        teleoperation_active = True
        print("Teleoperation activated")

    def stop_teleoperation() -> None:
        """
        Deactivate teleoperation control of the robot.

        Disables the application of teleoperation commands to the environment.

        Returns:
            None
        """
        nonlocal teleoperation_active
        teleoperation_active = False
        print("Teleoperation deactivated")

    teleoperation_callbacks: dict[str, Callable[[], None]] = {
        "R": reset_recording_instance,
        "START": start_teleoperation,
        "STOP": stop_teleoperation,
        "RESET": reset_recording_instance,
    }

    # Default to inactive without --xr and to the pipeline's configured default with --xr: a
    # headset drives START explicitly, but without one teleop is started locally just below
    # (``request_start``) so it still begins running -- flowing through the same state machine
    # keeps keyboard/host pause/resume working either way.
    teleoperation_active = env_cfg.isaac_teleop.teleoperation_active_default if args_cli.xr else False

    try:
        from isaaclab_teleop import create_isaac_teleop_device, poll_control_events

        teleop_interface = create_isaac_teleop_device(
            env_cfg.isaac_teleop,
            sim_device=args_cli.device,
            callbacks=teleoperation_callbacks,
            cloudxr_env_file=_resolve_cloudxr_env(args_cli.cloudxr_env, args_cli.xr),
            auto_launch_cloudxr=args_cli.auto_launch_cloudxr,
            enable_debug_visualization=args_cli.enable_debug_visualization,
            use_kit_xr_bridge=args_cli.xr,
            haptic_cfg=getattr(env_cfg, "haptic_feedback", None),
        )
    except Exception as e:
        logger.error(f"Failed to create teleop device: {e}")
        env.close()
        simulation_app.close()
        return

    print(f"Using teleop device: {teleop_interface}")

    # Optional controller haptics: no-ops unless the env declares a ``haptic_feedback`` config.
    haptic_update, haptic_stop = _make_haptic_io(env, teleop_interface, env_cfg)

    # Optional pollers for headset-free physical control (B/P/R, or right-button reset).
    # Advanced every frame in ``run_loop`` below -- never the primary device itself.
    control_pollers = _make_control_pollers(
        teleop_interface, env_cfg.isaac_teleop, app_launcher.has_window, args_cli.device
    )

    def run_loop():
        """Inner function to run the teleop loop with access to nonlocal variables."""
        nonlocal should_reset_recording_instance, teleoperation_active

        # reset environment
        env.reset()
        teleop_interface.reset()

        # Without --xr there is no headset to send START, so start locally ([B]/[P] can
        # still pause/resume). The reset() above is a host reset (a pure pulse), so it does
        # not cancel this start.
        if not args_cli.xr:
            teleop_interface.request_start()

        print("IsaacTeleop teleoperation started. Press 'R' to reset the environment.")

        # simulate environment
        while simulation_app.is_running():
            try:
                # run everything in inference mode
                with torch.inference_mode():
                    # get device command
                    action = teleop_interface.advance()
                    for poller in control_pollers:
                        poller.advance()

                    ctrl = poll_control_events(teleop_interface)
                    if ctrl.is_active is not None:
                        teleoperation_active = ctrl.is_active
                    if ctrl.should_reset:
                        should_reset_recording_instance = True

                    # action is None when IsaacTeleop session hasn't started yet
                    # (e.g. waiting for user to click "Start AR")
                    if action is None:
                        env.sim.render()
                        haptic_stop()
                    elif teleoperation_active:
                        # process actions
                        actions = action.repeat(env.num_envs, 1)
                        # apply actions
                        env.step(actions)
                        # render controller haptics from post-step contact forces
                        haptic_update()
                    else:
                        env.sim.render()
                        # not stepping: zero haptics so a paused grip stops buzzing
                        haptic_stop()

                    if should_reset_recording_instance:
                        env.reset()
                        teleop_interface.reset()
                        camera_feed_session.refresh()
                        should_reset_recording_instance = False
                        print("Environment reset complete")
            except Exception as e:
                logger.error(f"Error during simulation step: {e}")
                break

    # Run the teleoperation loop
    with teleop_interface, camera_feed_session.bind(env):
        run_loop()

    # close the simulator
    env.close()
    print("Environment closed")


if __name__ == "__main__":
    # run the main function
    main()
    # env.close() already closes the USD stage via sim.clear_instance().
    # Pump the event loop so the viewport processes closure, then close the app.
    simulation_app.update()
    simulation_app.close()
